# Proximal Policy Optimization for Pacman
import gymnasium as gym
import numpy as np

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
total_frames = 1_000_000

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
        max_steps=300
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


#######################################################
## Save model
#######################################################

torch.save({
    "policy": collector_policy_module.state_dict(),
    "value": value_module.state_dict(),
    "optimizer": optim.state_dict(),
}, "ppo_pacman.pt")


#######################################################
## Load model
#######################################################

checkpoint = torch.load("ppo_pacman.pt", map_location=device)

collector_policy_module.load_state_dict(checkpoint["policy"])
value_module.load_state_dict(checkpoint["value"])
optim.load_state_dict(checkpoint["optimizer"])

collector_policy_module.eval()
value_module.eval()

#######################################################
## Results
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
## Results on env1 to establish a baseline
#######################################################

def evaluate_ppo_agent(env, policy_module, num_episodes=100):
    returns = []
    successes = []
    final_scores = []
    percent_food_eaten = []
    normalized_scores = []
    episode_lengths = []

    policy_module.eval()

    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        for episode in range(num_episodes):
            td = env.reset()
            done = False
            episode_return = 0.0
            steps = 0
            last_info = None

            while not done:
                td = policy_module(td)
                td = env.step(td)

                reward = td["next", "reward"].item()
                episode_return += reward
                steps += 1

                terminated = td["next", "terminated"].item()
                truncated = td["next", "truncated"].item()
                done = terminated or truncated

                if done:
                    if "info" in td["next"].keys():
                        last_info = td["next", "info"]

                td = td["next"]

            returns.append(episode_return)
            episode_lengths.append(steps)

            if last_info is not None:
                successes.append(float(last_info.get("is_success", 0.0)))
                final_scores.append(float(last_info.get("final_score", 0.0)))
                percent_food_eaten.append(float(last_info.get("percent_food_eaten", 0.0)))
                normalized_scores.append(float(last_info.get("normalized_score", 0.0)))

    results = {
        "avg_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "win_rate": float(np.mean(successes)) if successes else 0.0,
        "avg_final_score": float(np.mean(final_scores)) if final_scores else 0.0,
        "avg_percent_food_eaten": float(np.mean(percent_food_eaten)) if percent_food_eaten else 0.0,
        "avg_normalized_score": float(np.mean(normalized_scores)) if normalized_scores else 0.0,
        "avg_episode_length": float(np.mean(episode_lengths)),
    }

    return results

results = evaluate_ppo_agent(test_env, collector_policy_module, num_episodes=100)

for k, v in results.items():
    print(f"{k}: {v}")





####################
def evaluate_ppo_agent_gym(base_env, num_episodes=100, device="cpu"):
    import numpy as np
    import torch
    from tensordict import TensorDict
    from torchrl.envs.utils import set_exploration_type, ExplorationType

    returns = []
    successes = []
    final_scores = []
    percent_food_eaten = []
    normalized_scores = []
    episode_lengths = []

    probabilistic_actor.eval()

    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        for _ in range(num_episodes):
            obs, info = base_env.reset()
            done = False
            ep_return = 0.0
            steps = 0
            last_info = {}

            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

                td = TensorDict(
                    {"observation": obs_tensor},
                    batch_size=[1],
                    device=device,
                )

                td = probabilistic_actor(td)

                action = td["action"].item()

                obs, reward, terminated, truncated, info = base_env.step(action)
                done = terminated or truncated

                ep_return += reward
                steps += 1
                last_info = info

            returns.append(ep_return)
            episode_lengths.append(steps)

            successes.append(float(last_info.get("is_success", 0.0)))
            final_scores.append(float(last_info.get("final_score", 0.0)))
            percent_food_eaten.append(float(last_info.get("percent_food_eaten", 0.0)))
            normalized_scores.append(float(last_info.get("normalized_score", 0.0)))

    return {
        "avg_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "win_rate": float(np.mean(successes)),
        "avg_final_score": float(np.mean(final_scores)),
        "avg_percent_food_eaten": float(np.mean(percent_food_eaten)),
        "avg_normalized_score": float(np.mean(normalized_scores)),
        "avg_episode_length": float(np.mean(episode_lengths)),
    }


env1_unwrapped = gym.make(
        'gymnasium_env/PacmanGen-v0',
        seed=0, 
        render_or_not=False, 
        render_mode="tinygrid",
        train_layouts=train_maps1, 
        test_layouts=test_maps, 
        split="train",
        max_steps=300
    )

results = evaluate_ppo_agent_gym(
    env1_unwrapped,
    num_episodes=100,
    device=device,
)

for k, v in results.items():
    print(f"{k}: {v}")