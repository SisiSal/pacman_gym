"""
"""

import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict

from .pacman.pacman import build_gym_args, ClassicGameRules
import numpy as np
from .pacman.layout import Layout, getLayout
import random as rd
import networkx as nx
import random
import math
from skimage.measure import block_reduce

class PacmanEnv(gym.Env):
    metadata = {"render_modes": ["human", "tinygrid", "gray", "dict", "state_pixels", "rgb_array"]}

    def __init__(
            self, seed, render_or_not, render_mode, move_ghosts=True, stochasticity=0.0,
            train_layouts=None, test_layouts=None, split="train", fixed_map=None, num_ghosts=3, max_steps=300
            ):
        """"""
        self._seed = seed
        self.np_random = np.random.default_rng(self._seed)

        # Rendering options
        self.beQuiet = not render_or_not
        self.render_or_not = render_or_not
        self.render_mode = render_mode

        # Rewards
        self.reward_goal = 10
        self.reward_crash = -10
        self.reward_food = 1
        self.reward_time = -0.1

        # Episode management
        self.max_steps = max_steps
        self.steps = 0
        self.history = []    

        # Additional options
        self.move_ghosts = move_ghosts
        self.stochasticity = stochasticity
        self.num_ghosts = num_ghosts

        # Map pools
        self.train_layouts = train_layouts or ["medium_01"]
        self.test_layouts = test_layouts or ["easy_01"]
        self.split = split
        self.fixed_map = fixed_map

        # Build the cyclic schedule (names)
        if self.fixed_map is not None:
            self.layout_cycle = [self.fixed_map]
        else:
            self.layout_cycle = self.train_layouts if self.split == "train" else self.test_layouts

        # Ensure layouts exist and are provided
        if len(self.layout_cycle) == 0:
            raise ValueError("No layouts provided for the selected split.")

        # Check that all layouts in the cycle exist
        missing = [name for name in self.layout_cycle if getLayout(name) is None]
        if missing:
            raise ValueError(
                "These layouts were not found in the layouts folder: "
                + ", ".join(missing)
            )

        # Start with the first layout in the cycle
        self._layout_idx = 0

        # Set the background image
        self.background_filename = "background.jpeg"

        self.grid_size = 1
        # Fix the observation space
        self.grid_height = 11
        self.grid_width = 19
        self.color_channels = 1

        self.height, self.width = 482, 482
        self.downsampling_size = 8

        # Default observation space (can be overridden by specific render modes)
        self.observation_space = Box(low=0, high=1, shape=(self.grid_height, self.grid_width), dtype=np.float32)

        # Action space: 0=Stop, 1=North, 2=South, 3=West, 4=East
        self.A = ["Stop", "North", "South", "West", "East"]
        self.action_space = Discrete(5) # default datatype is np.int64
        self.action_size = 5
        
        self.reward_range = (0, 10)

        # Set the observation space based on the rendering mode
        if self.render_mode == "tinygrid":
            self.observation_space = Box(
                low=0,
                high=1,
                shape=(
                    1,
                    self.grid_height * self.grid_size,
                    self.grid_width * self.grid_size,
                    ),
                dtype=np.float32,
            )
        elif self.render_mode == "gray":
            reduced_dim = math.ceil(self.height / self.downsampling_size)
            self.observation_space = Box(
                low=0,
                high=1,
                shape=(
                    reduced_dim, reduced_dim
                )
            )
        elif self.render_mode == "dict":
            reduced_dim = math.ceil(self.height / self.downsampling_size)
            self.observation_space = Dict({
                "gray": Box(
                    low=0,
                    high=1,
                    shape=(
                        reduced_dim, reduced_dim
                    ),
                    dtype=np.float32
                    ),
                "tinygrid": Box(
                    low=0,
                    high=1,
                    shape=(
                        self.grid_height * self.grid_size,
                        self.grid_width * self.grid_size
                    ),
                    dtype=np.float32
                )
            })
        elif self.render_mode == "state_pixels": # TODO
            reduced_dim = math.ceil(self.height / self.downsampling_size)
            self.observation_space = Box(
                low=0,
                high=1,
                shape=(
                    reduced_dim, reduced_dim
                )
            )
        elif self.render_mode == "rgb_array":
            self.observation_space = Box(
                low=0,
                high=255,
                shape=(self.height, self.width, 3),
                dtype=np.uint8
            )


    def step(self, action, observation_mode="human"):
        """
        Parameters
        ----------
        action :
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and terminated
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        agentIndex = 0
        self.steps += 1

        # Apply stochasticity to the action
        rdm = random.random()
        if rdm >= 2*self.stochasticity:
            action = [0, 1, 2, 3, 4][action]
        elif rdm >= self.stochasticity:
            action = [0, 3, 3, 1, 1][action]
        else:
            action = [0, 4, 4, 2, 2][action]


        if isinstance(action, np.int64) or isinstance(action, int):
            action = self.A[action]

        action = "Stop" if action not in self.get_legal_actions(0) else action

        # perform "doAction" for the pacman
        self.game.agents[agentIndex].doAction(self.game.state, action)
        self.game.take_action(agentIndex, action)

        if self.render_or_not and self.render_mode == "human":
            self.render("human")
        
        reward = self.game.state.data.scoreChange

        # move the ghosts
        if self.move_ghosts:
            if not self.game.gameOver:
                for agentIndex in range(1, len(self.game.agents)):
                    state = self.game.get_observation(agentIndex)
                    action = self.game.calculate_action(agentIndex, state)
                    self.game.take_action(agentIndex, action)
                    # Adjusted rendering
                    if self.render_or_not and self.render_mode == "human":
                        self.render("human")
                        
                    reward += self.game.state.data.scoreChange
                    if self.game.gameOver:
                        break
        
        # Check for termination and truncation
        terminated = self.game.gameOver 
        truncated = self._check_if_maxsteps()

        # Info can include whether max steps were used and whether the episode ended in 
        # success (win) or failure (lose)
        info = {}
        if terminated or truncated:
            info["maxsteps_used"] = truncated
            info["is_success"] = self.game.state.isWin()
            info["final_score"] = self.game.state.data.score
            info["layout_name"] = self.layout_name
            info["initial_num_food"] = int(self.initial_num_food)
            info["remaining_food"] = self.game.state.getNumFood()
            info["percent_food_eaten"] = float((self.initial_num_food - self.game.state.getNumFood())/max(1, self.initial_num_food))
            info["normalized_score"] = float(self.game.state.data.score / max(1, self.initial_num_food))
        observation = self.render(self.render_mode)

        # return self.game.state, reward, self.game.gameOver, dict()
        return observation, reward, terminated, truncated, info

    def reset(self, observation_mode="human", seed=None, options=None):
        super().reset(seed=seed)

        # Reset the environment to the next layout in the cycle
        self.steps = 0
        layout_name = self.layout_cycle[self._layout_idx]
        self.layout_name = layout_name
        self._layout_idx = (self._layout_idx + 1) % len(self.layout_cycle)

        # Build the game using the selected layout and parameters
        args = build_gym_args(
            layout_name=layout_name,
            num_ghosts=self.num_ghosts,
            be_quiet=self.beQuiet,
            zoom=1.0,
            frame_time=0.1,
            timeout=30,
            reward_goal=self.reward_goal,
            reward_crash=self.reward_crash,
            reward_food=self.reward_food,
            reward_time=self.reward_time,
        )

        # Pull objects from args
        self.layout = args["layout"]
        self.pacman = args["pacman"]
        self.ghosts = args["ghosts"]
        self.display = args["display"]
        self.numGames = args["numGames"]
        self.record = args["record"]
        self.numTraining = args["numTraining"]
        self.numGhostTraining = args["numGhostTraining"]
        self.withoutShield = args["withoutShield"]
        self.catchExceptions = args["catchExceptions"]
        self.timeout = args["timeout"]
        self.symX = args["symX"]
        self.symY = args["symY"]

        # Rules and game setup
        self.rules = ClassicGameRules(
            self.timeout,
            self.reward_goal,
            self.reward_crash,
            self.reward_food,
            self.reward_time,
        )

        if self.beQuiet:
            # Suppress output and graphics
            from .pacman import textDisplay

            self.gameDisplay = textDisplay.NullGraphics()
            self.rules.quiet = True
        else:
            self.gameDisplay = self.display
            self.rules.quiet = False

        # Create the game instance
        self.game = self.rules.newGame(
            self.layout,
            self.pacman,
            self.ghosts,
            self.gameDisplay,
            self.beQuiet,
            self.catchExceptions,
            self.symX,
            self.symY,
            self.background_filename
        )
        
        self.game.start_game()

        self.num_agents, self.num_food, self.non_wall_positions, self.wall_positions, self.all_edges = self.sample_prep(
            self.layout)
        self.initial_num_food = self.num_food
        
        mode = self.render_mode
        if self.beQuiet and mode in ("gray", "dict"):
            mode = "tinygrid"

        observation = self.render(mode)
        info = {"layout_name": layout_name, "num_food": int(self.num_food)}

        return observation, info

    def downsampling(self, x):
        dz = block_reduce(x, block_size=(self.downsampling_size, self.downsampling_size), func=np.mean)
        return dz

    def render(self, mode="human", close=False):

        if mode == "gray":
            img = self.game.compose_img(mode)
            return self.downsampling(img)
        elif mode == "human":
            return self.game.compose_img(mode)
        elif mode == "rgb_array":
            return self.game.compose_img(mode)
        elif mode == "tinygrid":
            obs = self.game.compose_img(mode="tinygrid").astype(np.float32)
            return obs[np.newaxis, :, :]
        elif mode == "dict":
            img = self.game.compose_img(mode)  # calls the fast renderer
            return {
                "gray" : self.downsampling(img),
                "tinygrid": self.game.compose_img(mode="tinygrid")
            }
        else:
            raise ValueError(f"Unsupported render mode: {mode}")



    def get_legal_actions(self, agentIndex):
        return self.game.state.getLegalActions(agentIndex)

    def get_action_lookup(self):
        return ACTION_LOOKUP

    def get_action_meanings(self):
        return self.A

    def _check_if_maxsteps(self):
        return (self.max_steps <= self.steps)

    @staticmethod
    def constraint_func(self):
        return

    def sample_prep(self, layout):
        width = layout.width
        height = layout.height
        num_food = np.count_nonzero(np.array(layout.food.data) == True)
        num_agents = len(layout.agentPositions)
        walls = str(layout.walls).split("\n")
        non_wall_positions = []
        wall_positions = []
        for r in range(height):
            for c in range(width):
                if walls[r][c] == 'F':
                    non_wall_positions.append((r, c))
                else:
                    wall_positions.append((r, c))

        def safe_add(s, p1, p2s, width, height):
            p1r, p1c = p1
            for p2 in p2s:
                p2r, p2c = p2
                if 0 <= p1r < height and 0 <= p2r < height and 0 <= p1c < width and 0 <= p2c < width:
                    s.add((p1, p2))

        edges = set()
        for r in range(height):
            for c in range(width):
                safe_add(edges, (r, c), [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)], width, height)

        return num_agents, num_food, non_wall_positions, wall_positions, edges

ACTION_LOOKUP = {
    0: 'stay',
    1: 'up',
    2: 'down',
    3: 'left',
    4: 'right',
}


