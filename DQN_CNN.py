### Deep Q-Learing with a CNN feature extractor for Pacman
import random
import gymnasium as gym
import numpy as np

from stable_baselines3 import DQN
from feature_extractors import policy_kwargs
from stable_baselines3.common.monitor import Monitor
from collections import defaultdict

from results import evaluate_dqn_by_layout, plot_layout_summary


######################################
## Create the environments
######################################
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
    env = gym.make(
        'gymnasium_env/PacmanGen-v0',
        seed=seed, 
        render_or_not=False, 
        render_mode="tinygrid",
        train_layouts=train_layouts, 
        test_layouts=test_layouts, 
        split=split,
        max_steps=600
    )
    return Monitor(env)

env1 = make_env(train_maps1, test_maps, split="train")
test_env1 = make_env(train_maps1, train_maps1+test_maps, split="test")

env2 = make_env(train_maps2, test_maps, split="train")
test_env2 = make_env(train_maps2, train_maps2+test_maps, split="test")

env3 = make_env(train_maps3, test_maps, split="train")
test_env3 = make_env(train_maps3, train_maps3+test_maps, split="test")

######################################
## Hyperparameter tuning
######################################

# def evaluate_dqn(model, env, n_eval_episodes=10):
#     rewards = []
#     successes = []

#     for _ in range(n_eval_episodes):
#         obs, _ = env.reset()
#         done = False
#         total_reward = 0
#         last_info = {}

#         while not done:
#             action, _ = model.predict(obs, deterministic=True)
#             obs, reward, terminated, truncated, info = env.step(action)

#             done = terminated or truncated
#             total_reward += reward
#             last_info = info

#         rewards.append(total_reward)
#         successes.append(float(last_info.get("is_success", 0.0)))

#     return {
#         "mean_reward": np.mean(rewards),
#         "std_reward": np.std(rewards),
#         "win_rate": np.mean(successes),
#     }

# def sample_hyperparams_DQN():
#     return {
#         "learning_rate": random.choice([0.0005, 0.00025, 0.0001, 0.00005]),
#         "buffer_size": random.choice([50000, 100000]),
#         "learning_starts": random.choice([100, 1000, 5000]),
#         "train_freq": random.choice([1, 4, 8]),
#         "gradient_steps": random.choice([1, 4, 8]),
#         "target_update_interval": random.choice([100, 500, 1000]),
#         "gamma": random.choice([0.95, 0.99, 0.995]),
#         "exploration_fraction": random.choice([0.1, 0.2, 0.3]),
#         "exploration_final_eps": random.choice([0.01, 0.05, 0.1]),
#     }

# def run_trial(trial_id, total_timesteps=100_000):

#     params = sample_hyperparams_DQN()

#     model = DQN(
#         policy="CnnPolicy",
#         env=env1,
#         policy_kwargs=policy_kwargs,
#         verbose=0,
#         **params,
#     )

#     model.learn(total_timesteps=total_timesteps)

#     stats = evaluate_dqn(
#         model,
#         env1,
#         n_eval_episodes=10,
#     )

#     env1.close()

#     return {
#         "trial_id": trial_id,
#         "params": params,
#         "mean_reward": stats["mean_reward"],
#         "std_reward": stats["std_reward"],
#         "win_rate": stats["win_rate"],
#     }

# results = []
# n_trials = 15


# for i in range(n_trials):
#     result = run_trial(i, total_timesteps=100_000)
#     results.append(result)
#     print(
#         f"Trial {i}: "
#         f"mean_reward={result['mean_reward']:.2f}, "
#         f"win_rate={result['win_rate']:.2f}, "
#         f"params={result['params']}"
#         )
    
# best_result = max(results, key=lambda x: x["mean_reward"])
# print("\nBest trial:")
# print(best_result)


######################################
## Train the DQN agent
######################################

import gc
gc.collect()

#tensorboard --logdir .\tensorboard_logs
model = DQN('CnnPolicy', 
            env3, 
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=0.00025,
            buffer_size=50000,
            learning_starts=1000,
            train_freq=4,
            gradient_steps=4,
            target_update_interval=1000,
            gamma=0.95,
            exploration_initial_eps=1.0,
            exploration_fraction=0.1,
            exploration_final_eps=0.05,
            tensorboard_log="./tensorboard_logs/")

model.learn(total_timesteps=1682895, #~3mil steps for ~25k episodes
            log_interval=1000,
            tb_log_name="dqn_env3_3conv")

model.save("dqn_env3_3conv")

######################################
## Test the DQN agent
######################################

del model

model_env1 = DQN.load("dqn_env1_3conv", env=env1)
model_env2 = DQN.load("dqn_env2_3conv", env=env2)
model_env3 = DQN.load("dqn_env3_3conv", env=env3)

results_by_layout_env1, summary_by_layout_env1 = evaluate_dqn_by_layout(model_env1, test_env1, n_eval_episodes=600, print_results=True)
results_by_layout_env2, summary_by_layout_env2 = evaluate_dqn_by_layout(model_env2, test_env2, n_eval_episodes=1000, print_results=True)
results_by_layout_env3, summary_by_layout_env3 = evaluate_dqn_by_layout(model_env3, test_env3, n_eval_episodes=00, print_results=True)

######################################
## Plot results
######################################

df_summary_env1 = plot_layout_summary(summary_by_layout_env1, test_maps=test_maps)
df_summary_env2 = plot_layout_summary(summary_by_layout_env2, test_maps=test_maps)
df_summary_env3 = plot_layout_summary(summary_by_layout_env3, test_maps=test_maps)

print(df_summary_env1)
df_summary_env1.to_csv("layout_summary_env1.csv", index=False)

print(df_summary_env2)
df_summary_env2.to_csv("layout_summary_env2.csv", index=False)

print(df_summary_env3)
df_summary_env3.to_csv("layout_summary_env3.csv", index=False)
