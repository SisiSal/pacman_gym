### Deep Q-Learing with a CNN feature extractor for Pacman
import random
import gymnasium as gym

from stable_baselines3 import DQN
from feature_extractors import policy_kwargs
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from collections import defaultdict


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
        max_steps=300
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

def sample_hyperparams_DQN():
    return {
        "learning_rate": 10 ** random.uniform(-5, -3),
        "buffer_size": random.choice([50000, 100000]),
        "learning_starts": random.choice([100, 1000, 5000]),
        "train_freq": random.choice([1, 4, 8]),
        "gradient_steps": random.choice([1, 4, 8]),
        "target_update_interval": random.choice([100, 500, 1000]),
        "gamma": random.choice([0.95, 0.99, 0.995]),
        "exploration_fraction": random.choice([0.1, 0.2, 0.3]),
        "exploration_final_eps": random.choice([0.01, 0.05, 0.1]),
    }

def run_trial(trial_id, total_timesteps=100_000):

    params = sample_hyperparams_DQN()

    model = DQN(
        policy="CnnPolicy",
        env=env1,
        policy_kwargs=policy_kwargs,
        verbose=0,
        **params,
    )

    model.learn(total_timesteps=total_timesteps)

    mean_reward, std_reward = evaluate_policy(
        model,
        env1,
        n_eval_episodes=10,
        deterministic=True,
    )

    env1.close()

    return {
        "trial_id": trial_id,
        "params": params,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
    }

results = []
n_trials = 20


for i in range(n_trials):
    result = run_trial(i, total_timesteps=100_000)
    results.append(result)
    print(f"Trial {i}: mean_reward={result['mean_reward']:.2f}, params={result['params']}")

best_result = max(results, key=lambda x: x["mean_reward"])
print("\nBest trial:")
print(best_result)


######################################
## Train the DQN agent
######################################

model = DQN('CnnPolicy', 
            env1, 
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=0.0007,
            buffer_size=50000,
            learning_starts=100,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            gamma=0.99,
            exploration_initial_eps=1.0,
            exploration_fraction=0.3,
            exploration_final_eps=0.1,
            tensorboard_log="./tensorboard_logs/")

model.learn(total_timesteps=3000000, #~3mil steps for ~25k episodes
            log_interval=10,
            tb_log_name="dqn_easy_01")

model.save("dqn_easy_01_3conv")

######################################
## Test the DQN agent
######################################

del model

env = test_env1

model = DQN.load("dqn_easy_01_2conv", env=env)

# store one record per episode, grouped by layout name
results_by_layout = defaultdict(list)

n_eval_episodes = 600

for episode in range(n_eval_episodes):
    obs, info = env.reset()
    done = False
    episode_reward = 0.0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated

    layout_name = info["layout_name"]

    results_by_layout[layout_name].append({
        "episode_reward": episode_reward,
        "is_success": info.get("is_success", False),
        "final_score": info.get("final_score", 0.0),
        "maxsteps_used": info.get("maxsteps_used", False),
        "initial_num_food": info.get("initial_num_food", 0),
        "remaining_food": info.get("remaining_food", 0),
        "percent_food_eaten": info.get("percent_food_eaten", 0.0),
        "normalized_score": info.get("normalized_score", 0.0),
    })

env.close()

import numpy as np

def summarize_results(results_by_layout):
    summary_by_layout = {}

    for layout_name, episodes in results_by_layout.items():
        rewards = [ep["episode_reward"] for ep in episodes]
        wins = [ep["is_success"] for ep in episodes]
        final_scores = [ep["final_score"] for ep in episodes]
        food_eaten = [ep["percent_food_eaten"] for ep in episodes]
        normalized_scores = [ep["normalized_score"] for ep in episodes]
        maxsteps_used = [ep["maxsteps_used"] for ep in episodes]

        summary_by_layout[layout_name] = {
            "num_episodes": len(episodes),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "win_rate": float(np.mean(wins)),
            "mean_final_score": float(np.mean(final_scores)),
            "mean_percent_food_eaten": float(np.mean(food_eaten)),
            "mean_normalized_score": float(np.mean(normalized_scores)),
            "maxsteps_rate": float(np.mean(maxsteps_used))
            }


    for layout_name, stats in summary_by_layout.items():
        print(f"\nLayout: {layout_name}")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    
    return summary_by_layout

summary1 = summarize_results(results_by_layout)
