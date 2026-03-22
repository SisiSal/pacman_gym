### Deep Q-Learing with Value Function Approximation for Pacman
import gymnasium as gym
import pacman_gym
import torch as th
import torch.nn as nn

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

#####################################
## Create the feature extractor
#####################################
class SmallPacmanCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]   # should be 1

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
    
policy_kwargs = dict(
    features_extractor_class=SmallPacmanCNN,
    features_extractor_kwargs=dict(features_dim=128),
)

######################################
## Create the environment
######################################
train_maps1 = ["medium_01"]

train_maps2 = ["medium_01", "medium_02",
              "hard_01", "hard_02"]

train_maps3 = ["medium_01", "medium_02", "medium_03", "medium_04",
              "hard_01", "hard_02", "hard_03", "hard_04"]

test_maps  = ["easy_01", "easy_02",
              "medium_05", "medium_06",
              "hard_05", "hard_06"]


env = gym.make(
    'gymnasium_env/PacmanGen-v0',
    seed=0, 
    render_or_not=False, 
    render_mode="tinygrid",
    train_layouts=train_maps1, 
    test_layouts=test_maps, 
    split="train"
)

######################################
## Train the DQN agent
######################################
model = DQN('CnnPolicy', 
            env, 
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=1e-4,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.1,     # more exploration later
            exploration_fraction=0.3,      # explore for longer
            tensorboard_log="./tensorboard_logs/")
model.learn(total_timesteps=100000, 
            log_interval=10,
            tb_log_name="dqn_medium_01")
model.save("dqn_medium_01")

del model

model = DQN.load("dqn_medium_01", env=env)

obs, info = env.reset()
print(f"Initial observation shape: {obs.shape}, observation space: {env.observation_space}, info: {info}")
for episode in range(5):
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    print(f"Episode {episode + 1} finished info: {info}")
    obs, info = env.reset()

env.close

######################################
## Train the PPO agent
######################################

model = PPO('CnnPolicy', 
            env,
            policy_kwargs=policy_kwargs,
            verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("ppo_pacman")