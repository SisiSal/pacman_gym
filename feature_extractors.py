### Custom feature extractors
import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from pacman_gym.envs.pacman.game import Directions, Actions
import pacman_gym.envs.pacman.util as utils
########################################
## Custom feature extractor for DQN
########################################

class SmallPacmanCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0] # number of channels

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        # )


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
    features_extractor_kwargs=dict(features_dim=256),
)

#################################################
## Feature extractor for Approximate Q-Learning
#################################################

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
        """
        utils.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        """
        Returns a feature vector with one feature for each (state,action) pair.
        """
        feats = utils.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        """
        Breaks the state down into x and y coordinates and creates features for each
        """
        feats = utils.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[1]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats
    
def closestFood(pos, food, walls):
    """
    Returns the distance to the closest food using BFS
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()

    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

def closestTarget(pos, targets, walls):
    """
    Returns the distance to the closest target using BFS
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    targets = set(targets)

    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        
        # if we find a target at this location then exit
        if (pos_x, pos_y) in targets:
            return dist
        
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))

    # no target found
    return None

class AdvancedExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - distance to the closest food
    - whether a capsule will be eaten
    - distance to the closest capsule
    - whether a ghost collision is imminent
    - whether an active ghost is one step away
    - whether a scared ghost is one or two steps away
    - distance to the closest active ghost
    - inverse of the distance to the closest active ghost (as a measure of risk)
    - distance to the closest scared ghost
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        curr_food_count = state.getNumFood()
        walls = state.getWalls()
        capsules = state.getCapsules()
        ghost_states = state.getGhostStates() # to get ghost positions and scared times

        features = utils.Counter()

        # Feature 1: bias, always 1.0 
        # Acts as a bias term for the weights.
        # This allows the agent to learn a baseline value 
        # for taking an action in any state, 
        # even if all other features are zero.
        features["bias"] = 1.0

        # current pacman position
        x, y = state.getPacmanPosition()

        # next pacman position after he takes the action
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # food count after he takes the action
        next_food_count = curr_food_count - (1 if food[next_x][next_y] else 0)

        # Split the ghost states into scared and active ghosts
        active_ghosts = []
        scared_ghosts = []

        # Get positions of active and scared ghosts
        for ghost in ghost_states:
            ghost_pos = ghost.getPosition()
            if ghost_pos is not None:
                ghost_pos = (int(ghost_pos[0]), int(ghost_pos[1]))
                if ghost.scaredTimer > 0:
                    scared_ghosts.append(ghost_pos)
                else:
                    active_ghosts.append(ghost_pos)

        # Feature 2: number of active ghosts 1-step away
        features["#-of-active-ghosts-1-step-away"] = sum(
            (next_x, next_y) in Actions.getLegalNeighbors(g, walls) 
            for g in active_ghosts
            )

        # Feature 3: number of scared ghosts 1-or-2-step away
        scared_count = 0
        for g in scared_ghosts:
            d = closestTarget((next_x, next_y), [g], walls)
            if d is not None and d <= 2:
                scared_count += 1
        features["#-of-scared-ghosts-1-or-2-step-away"] = scared_count

        # Feature 4: if there is no danger of ghosts then eat food
        if not features["#-of-active-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0
        
        # Feature 5: normalized eaten food count
        features["food-cleared"] = 1.0 - float(next_food_count) / (walls.width * walls.height)

        # Feature 6: distance to the closest food
        dist_food = closestFood((next_x, next_y), food, walls)
        if dist_food is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist_food) / (walls.width * walls.height)

        # Feature 7: distance to the closest capsule
        dist_capsule = closestTarget((next_x, next_y), capsules, walls)
        if dist_capsule is not None:
            features["closest-capsule"] = float(dist_capsule) / (walls.width * walls.height)

        # Feature 8 and 9: if there is danger of ghosts then go towards nearest capsule and eat it
        if features["#-of-active-ghosts-1-step-away"]:
            if (next_x, next_y) in capsules:
                features["eats-capsule"] = 1.0
            # if dist_capsule is not None:
            #     features["towards-capsule"] = 1.0/(1.0 + float(dist_capsule))

        # Feature 10 and 11: distance and inverse distance to the closest active ghost
        dist_active_ghost = closestTarget((next_x, next_y), active_ghosts, walls)
        if dist_active_ghost is not None:
            features["closest-active-ghost"] = float(dist_active_ghost) / (walls.width * walls.height)
            #features["inv-closest-active-ghost"] = 1.0 / (1.0 + float(dist_active_ghost))

        # Feature 12: distance to the closest scared ghost
        dist_scared_ghost = closestTarget((next_x, next_y), scared_ghosts, walls)
        if dist_scared_ghost is not None:
            features["closest-scared-ghost"] = float(dist_scared_ghost) / (walls.width * walls.height)

        features.divideAll(10.0)
        return features