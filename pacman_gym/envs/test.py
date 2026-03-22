import gymnasium as gym
from pacman_gym.envs.pacmanInterface import PacmanEnv

train_maps = ["easy_01", "easy_02", "medium_01", "medium_02", 
              "medium_03", "medium_04", "medium_05", "medium_06",
              "hard_01", "hard_02", "hard_03", "hard_04", "hard_05", "hard_06"]
test_maps  = []

#testing the human rendering mode
env_train = PacmanEnv(seed=0, render_or_not=True, render_mode="human",
                      train_layouts=train_maps, test_layouts=test_maps, split="train", max_steps=200)


obs, info = env_train.reset()
print("RESET INFO:", info)

done = False

while not done:
    a = env_train.action_space.sample()
    obs, reward, terminated, truncated, info = env_train.step(a)
    done = terminated or truncated

    if done:
        print("\nEPISODE FINISHED")
        print("INFO:", info)

# normalize reward after episode ends
env_train.close()


#######################################################
# testing the tinygrid rendering mode
env_train = PacmanEnv(seed=0, render_or_not=False, render_mode="tinygrid",
                      train_layouts=train_maps, test_layouts=test_maps, split="train", max_steps=10)

obs, info = env_train.reset()
done = False
t = 0
while not done:
    print(f"\nStep {t}")
    print(obs)
    print(set(obs.flatten()))
    a = env_train.action_space.sample()
    obs, reward, terminated, truncated, info = env_train.step(a)
    done = terminated or truncated
    t += 1

env_train.close()



#######################################################
# testing human rendering mode with keyboard input
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