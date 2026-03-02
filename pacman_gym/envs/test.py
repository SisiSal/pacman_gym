import gymnasium as gym
from pacman_gym.envs.pacmanInterface import PacmanEnv

train_maps = ["train_simple_01", "train_regular_01", "train_hard_01"]
test_maps  = ["train_simple_02", "train_regular_02", "train_hard_02"]

#testing the human rendering mode
env_train = PacmanEnv(seed=0, render_or_not=True, render_mode="human",
                      train_layouts=train_maps, test_layouts=test_maps, split="train")


obs, info = env_train.reset()

done = False
while not done:
    a = env_train.action_space.sample()
    obs, reward, terminated, truncated, info = env_train.step(a)
    done = terminated or truncated

env_train.close()


# #testing the tinygrid rendering mode
# env_train = PacmanEnv(seed=0, render_or_not=False, render_mode="tinygrid",
#                       train_layouts=train_maps, test_layouts=test_maps, split="train")

# obs, info = env_train.reset()
# done = False
# t = 0
# while not done:
#     print(f"\nStep {t}")
#     print(obs)  # shows tinygrid observation (whatever type it is: array/list/string)
#     a = env_train.action_space.sample()
#     obs, reward, terminated, truncated, info = env_train.step(a)
#     done = terminated or truncated
#     t += 1

# env_train.close()

# from PIL import Image
# img = Image.open("pacman_gym/envs/pacman/imgs/agent.png")
# print(img.size)