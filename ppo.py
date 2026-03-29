# Proximal Policy Optimization for Pacman
import gymnasium as gym

import warnings
warnings.filterwarnings("ignore")
from torch import multiprocessing


from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv, GymWrapper)
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, ValueOperator
from torch.distributions import Categorical
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

import pacman_gym
#######################################################
## Define hyperparameters
#######################################################

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)

# number of cells in each layer i.e. output dim.
num_cells = 128  

# try lr = 1e-3, 3e-4, 1e-4
lr = 3e-4

max_grad_norm = 1.0

#######################################################
## Data collection parameters
#######################################################

#try 1000 2000 4000
frames_per_batch = 4000

# For a complete training, bring the number of frames up to 1M
total_frames = 500_000

#######################################################
## PPO parameters
#######################################################

# try 64 128 256 (match to frames_per_batch)
sub_batch_size = 256  # cardinality of the sub-samples gathered from the current data in the inner loop

# optimization steps per batch of data collected
# try 5 10
num_epochs = 5  

# clip value for PPO loss
clip_epsilon = (
    0.2 
)

gamma = 0.99
lmbda = 0.95

# Coefficient for the entropy bonus
# can help exploration and stabilize training
entropy_eps = 0.02  

#######################################################
## Define environment, Transforms, and Normalization
#######################################################

train_maps1 = ["easy_01"]

train_maps2 = ["easy_01", "medium_01", "medium_02",
              "hard_01", "hard_02"]

train_maps3 = ["easy_01",
               "medium_01", "medium_02", "medium_03", "medium_04",
              "hard_01", "hard_02", "hard_03", "hard_04"]

test_maps  = ["easy_02",
              "medium_05", "medium_06",
              "hard_05", "hard_06"]

def make_env(train_layouts, test_layouts, split, seed=0):
    base_env = gym.make(
        'gymnasium_env/PacmanGen-v0',
        seed=seed, 
        render_or_not=False, 
        render_mode="tinygrid",
        train_layouts=train_layouts, 
        test_layouts=test_layouts, 
        split=split,
        max_steps=600
    )
    env_torchRL = GymWrapper(base_env, device=device, categorical_action_encoding=True)
    env = TransformedEnv(
        env_torchRL,
        Compose(
            # normalize observations
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(),
            StepCounter(),
        ),
    )
    env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
    
    # print("normalization constant shape:", env.transform[0].loc.shape)
    # print("observation_spec:", env.observation_spec)
    # print("reward_spec:", env.reward_spec)
    # print("input_spec:", env.input_spec)
    # print("action_spec (as defined by input_spec):", env.action_spec)
    
    # check_env_specs(env)

    rollout = env.rollout(3)
    # print("rollout of three steps:", rollout)
    # print("Shape of the rollout TensorDict:", rollout.batch_size)

    return env

env1 = make_env(train_maps1, test_maps, split="train")
env2 = make_env(train_maps2, test_maps, split="train", seed=1)
env3 = make_env(train_maps3, test_maps, split="train", seed=2)

test_env = make_env(train_maps1, train_maps1, split="test", seed=3)

test_env1 = make_env(train_maps1, test_maps, split="test", seed=4)
test_env2 = make_env(train_maps2, test_maps, split="test", seed=5)
test_env3 = make_env(train_maps3, test_maps, split="test", seed=6)

#######################################################
## Policy
#######################################################

env = env1

actor_net = nn.Sequential(
    nn.Flatten(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(env.action_spec.space.n, device=device), # logits for 5 actions
)


actor_module = TensorDictModule(
    actor_net, in_keys=["observation"], out_keys=["logits"]
)

probabilistic_actor = ProbabilisticActor(
    module=actor_module,
    spec=env.action_spec,
    in_keys=["logits"],
    distribution_class=Categorical,
    return_log_prob=True,
    # log-prob for the numerator of the importance weights
)

class SqueezeAction(torch.nn.Module):
    def forward(self, action):
        return action.squeeze(-1)

squeeze_action_module = TensorDictModule(
    module=SqueezeAction(),
    in_keys=["action"],
    out_keys=["action"],
)

collector_policy_module = TensorDictSequential(
    probabilistic_actor,   
    squeeze_action_module,
)

# td = env.reset()
# out = collector_policy_module(td)
# print(out["action"].shape)
# print(out["action"])
# print(out["logits"].shape)

#######################################################
## Value network
#######################################################

value_net = nn.Sequential(
    nn.Flatten(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(1, device=device), 
)

value_module = ValueOperator(
    module=value_net,
    in_keys=["observation"],
)

print("Running policy:", collector_policy_module(env.reset()))
print("Running value:", value_module(env.reset()))

#######################################################
## Data collector
#######################################################

collector = SyncDataCollector(
    env,
    collector_policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
)


#######################################################
## Loss function
#######################################################

advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True, device=device,
)

loss_module = ClipPPOLoss(
    actor_network=probabilistic_actor,
    critic_network=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coeff=entropy_eps,
    critic_coeff=1.0,
    loss_critic_type="smooth_l1",
)

optim = torch.optim.Adam(loss_module.parameters(), lr)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
)

#######################################################
## Training loop
#######################################################

logs = defaultdict(list)
pbar = tqdm(total=total_frames)
eval_str = ""

# Iterate over the collector until it reaches the total number of frames
for i, tensordict_data in enumerate(collector):

    for _ in range(num_epochs):
        advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        perm = torch.randperm(data_view.shape[0], device=data_view.device)

        for start in range(0, frames_per_batch, sub_batch_size):
            idx = perm[start:start + sub_batch_size]
            subdata = data_view[idx]
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            optim.zero_grad()

            # Optimization: backward, grad clipping and optimization step
            loss_value.backward()
            # Keep gradient norm bounded
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()

    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    pbar.update(tensordict_data.numel())
    cum_reward_str = (
        f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
    )
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs["lr"].append(optim.param_groups[0]["lr"])
    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
    if i % 10 == 0:
        # Evaluate the policy once every 10 batches of data.
        # Evaluation: execute the policy without exploration for a given number of steps.
        # The ``rollout`` method of the ``env`` can take a policy as argument:
        # it will then execute this policy at each step.
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            # execute a rollout with the trained policy
            eval_rollout = test_env.rollout(300, collector_policy_module)
            logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
            logs["eval reward (sum)"].append(
                eval_rollout["next", "reward"].sum().item()
            )
            logs["eval step_count"].append(eval_rollout["step_count"].max().item())
            eval_str = (
                f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                f"eval step-count: {logs['eval step_count'][-1]}"
            )
            del eval_rollout
    pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

    # Learning rate scheduler
    scheduler.step()


# td = env.reset()
# out = collector_policy_module(td)
# print(out["action"].shape)
# print(out["action"])

# td = env.reset()
# out = collector_policy_module(td)
# out["action"] = out["action"].squeeze(-1)
# env.step(out)

# td = test_env.reset()
# done = False
# step = 0

# while not done and step < 10:
#     out = collector_policy_module(td)
#     print("step", step)
#     print("chosen index:", out["action"].item())
#     print("logits:", out["logits"])

#     td = test_env.step(out)

#     reward = td["next", "reward"].item()
#     terminated = td["next", "terminated"].item()
#     truncated = td["next", "truncated"].item()
#     done = terminated or truncated

#     print("reward:", reward, "terminated:", terminated, "truncated:", truncated)
#     print()

#     td = td["next"]
#     step += 1


#######################################################
## Results on training maps
#######################################################

plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.plot(logs["reward"])
plt.title("training rewards (average)")
plt.subplot(2, 2, 2)
plt.plot(logs["step_count"])
plt.title("Max step count (training)")
plt.subplot(2, 2, 3)
plt.plot(logs["eval reward (sum)"])
plt.title("Return (test)")
plt.subplot(2, 2, 4)
plt.plot(logs["eval step_count"])
plt.title("Max step count (test)")
plt.show()

#######################################################
## Results on testing maps
#######################################################

import numpy as np

# def evaluate_test_maps_by_name(test_env, policy_module, num_episodes=20):
#     returns_by_map = defaultdict(list)
#     steps_by_map = defaultdict(list)

#     with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
#         for _ in range(num_episodes):
#             td = test_env.reset()

#             # layout name comes from reset info
#             layout_name = td["layout_name"] if "layout_name" in td.keys() else None

#             done = False
#             episode_return = 0.0
#             step_count = 0

#             while not done:
#                 td = policy_module(td)
#                 td = test_env.step(td)

#                 reward = td["next", "reward"].item()
#                 episode_return += reward

#                 terminated = td["next", "terminated"].item()
#                 truncated = td["next", "truncated"].item()
#                 done = terminated or truncated

#                 step_count += 1
#                 td = td["next"]

#             returns_by_map[layout_name].append(episode_return)
#             steps_by_map[layout_name].append(step_count)


#     summary = {}
#     for map_name in returns_by_map:
#         summary[map_name] = {
#             "mean_return": float(np.mean(returns_by_map[map_name])),
#             "std_return": float(np.std(returns_by_map[map_name])),
#             "mean_steps": float(np.mean(steps_by_map[map_name])),
#             "n_episodes": len(returns_by_map[map_name]),
#         }

#     return summary

results_by_map = defaultdict(lambda: {
    "returns": [],
    "normalized_scores": [],
    "percent_food_eaten": [],
    "successes": [],
})

for episode in range(500):
    td = test_env.reset()
    done = False
    episode_return = 0.0

    while not done:
        td = policy_module(td)
        td = test_env.step(td)

        reward = td["next", "reward"].item()
        episode_return += reward

        terminated = td["next", "terminated"].item()
        truncated = td["next", "truncated"].item()
        done = terminated or truncated

        if done:
            info = td["next", "info"]   # assuming terminal info is accessible
            layout_name = info["layout_name"]

            results_by_map[layout_name]["returns"].append(episode_return)
            results_by_map[layout_name]["normalized_scores"].append(info["normalized_score"])
            results_by_map[layout_name]["percent_food_eaten"].append(info["percent_food_eaten"])
            results_by_map[layout_name]["successes"].append(float(info["is_success"]))
            # inspect terminal info to see if we can get layout names and other useful info for evaluation
            print(td)

        td = td["next"]

summary = {}

for layout_name, vals in results_by_map.items():
    summary[layout_name] = {
        "mean_return": float(np.mean(vals["returns"])),
        "mean_normalized_score": float(np.mean(vals["normalized_scores"])),
        "mean_percent_food_eaten": float(np.mean(vals["percent_food_eaten"])),
        "win_rate": float(np.mean(vals["successes"])),
        "num_episodes": len(vals["returns"]),
    }