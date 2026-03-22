import gymnasium as gym
from pacman_gym.envs.pacmanInterface import PacmanEnv

train_maps = ["easy_01", "easy_02", "easy_01", "easy_02", 
              "medium_03", "medium_04", "medium_05", "medium_06",
              "train_hard_01", "train_hard_02", "train_hard_03", "train_hard_04", "train_hard_05", "train_hard_06"]
test_maps  = ["medium_06"]

#testing the human rendering mode
env_train = PacmanEnv(seed=0, render_or_not=True, render_mode="human",
                      train_layouts=train_maps, test_layouts=test_maps, split="test", max_steps=1000)


obs, info = env_train.reset()

total_reward = 0
done = False

while not done:
    a = env_train.action_space.sample()
    obs, reward, terminated, truncated, info = env_train.step(a)
    total_reward += reward
    done = terminated or truncated

# normalize reward after episode ends
num_food = info["num_food"]
normalized_reward = total_reward / num_food  # example normalization
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


#####testing human rendering mode with keyboard input
key_to_action = {
    "w": 1,  # North
    "s": 2,  # South
    "a": 3,  # West
    "d": 4,  # East
    " ": 0   # Stop
}

obs, info = env_train.reset()

done = False
while not done:
    key = input("Move (w/a/s/d): ")
    action = key_to_action.get(key, 0)

    obs, reward, terminated, truncated, info = env_train.step(action)
    done = terminated or truncated