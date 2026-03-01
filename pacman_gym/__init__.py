from gymnasium.envs.registration import register

register(
    id='gymnasium_env/PacmanGen-v0',
    entry_point='pacman_gym.envs:PacmanEnv',
)