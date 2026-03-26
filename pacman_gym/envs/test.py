import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from pacman_gym.envs.pacmanInterface import PacmanEnv

all_layouts = ["hard_03", "hard_05"]
train_maps = ["easy_01", 
              "medium_01", "medium_02", "medium_03", "medium_04",
              "hard_01", "hard_02", "hard_03", "hard_04"]
test_maps  = ["easy_02",
              "medium_05", "medium_06",
              "hard_05", "hard_06"]

###########################################################
#testing the human rendering mode
env_train = PacmanEnv(seed=0, render_or_not=True, render_mode="human",
                      train_layouts=all_layouts, test_layouts=test_maps, split="train", max_steps=200)


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


#######################################################
# testing the rgb_array rendering mode

env = PacmanEnv(
    seed=0,
    render_or_not=False,
    render_mode="rgb_array",
    train_layouts=train_maps,
    test_layouts=test_maps,
    split="train",
)

obs, info = env.reset()

print("reset type:", type(obs))
print("reset dtype:", obs.dtype)
print("reset shape:", obs.shape)
print("min/max:", obs.min(), obs.max())
print("pacman:", env.game.state.getPacmanPosition())
print("ghosts:", env.game.state.getGhostPositions())

plt.imsave("rgb_reset.png", obs)

done = False
step_num = 0
prev_obs = obs

plt.imshow(obs)
plt.axis("off")
plt.show()

while not done and step_num < 5:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    print(
        f"step {step_num}: reward={reward}, "
        f"pacman={env.game.state.getPacmanPosition()}, "
        f"ghosts={env.game.state.getGhostPositions()}, "
        f"shape={obs.shape}, dtype={obs.dtype}"
    )

    diff = np.mean(np.abs(obs.astype(np.float32) - prev_obs.astype(np.float32)))
    print("mean pixel diff:", diff)

    plt.imsave(f"rgb_step_{step_num}.png", obs)

    prev_obs = obs
    done = terminated or truncated
    step_num += 1

env.close()