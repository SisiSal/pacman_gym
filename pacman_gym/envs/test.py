import gymnasium as gym
from pacman_gym.envs.pacmanInterface import PacmanEnv

# env = PacmanEnv(seed=42, render_or_not=True, render_mode="grey")
# obs = env.reset()
# done = False

# while not done:
#     action = env.action_space.sample()  # random action
#     obs, reward, terminated,  info = env.step(action)
#     done = terminated
#     env.render()

# env.close()

train_maps = ["train_simple_01", "train_regular_01", "train_hard_01", ...]
test_maps  = ["test_01_easy", "test_02", ..., "test_08_hard"]

env_train = PacmanEnv(seed=0, render_or_not=False, render_mode="gray",
                      move_ghosts=True,
                      train_layouts=train_maps, test_layouts=test_maps,
                      split="train")

env_test = PacmanEnv(seed=0, render_or_not=True, render_mode="human",
                     move_ghosts=True,
                     train_layouts=train_maps, test_layouts=test_maps,
                     split="test")



