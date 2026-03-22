### Deep Q-Learing with Value Function Approximation for Pacman
import gymnasium as gym
from stable_baselines3 import DQN

env = gym.make('PacmanGen-v0')

model = DQN('MlpPolicy', env, verbose=1)