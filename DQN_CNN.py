### Deep Q-Learing with a CNN feature extractor for Pacman
import random
import gymnasium as gym
import pacman_gym

from stable_baselines3 import DQN, PPO
from feature_extractors import policy_kwargs
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

######################################
## Create the environment
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
env2 = make_env(train_maps2, test_maps, split="train")
env3 = make_env(train_maps3, test_maps, split="train")

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
            exploration_fraction=0.3,
            exploration_final_eps=0.1,
            tensorboard_log="./tensorboard_logs/")

model.learn(total_timesteps=3000000, #~3mil steps for ~25k episodes
            log_interval=10,
            tb_log_name="dqn_easy_01")

model.save("dqn_easy_01")

#del model

#model = DQN.load("dqn_easy_01", env=env1)

obs, info = env1.reset()
print(f"Initial observation shape: {obs.shape}, observation space: {env1.observation_space}, info: {info}")
for episode in range(5):
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env1.step(action)
        done = terminated or truncated
    
    print(f"Episode {episode + 1} finished info: {info}")
    obs, info = env1.reset()

env1.close

######################################
## Train the PPO agent
######################################

model = PPO('CnnPolicy', 
            env1,
            policy_kwargs=policy_kwargs,
            verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("ppo_pacman")