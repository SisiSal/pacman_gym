# Approximate Q-learning implementation for PacmanGen environment

import gymnasium as gym
import pacman_gym
import numpy as np
import random
from collections import defaultdict

from feature_extractors import AdvancedExtractor
from pacman_gym.envs.pacman.util import Counter

from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
from feature_extractors import FeatureExtractor

train_maps1 = ["easy_01"]

train_maps2 = ["easy_01", "medium_01", "medium_02",
              "hard_01", "hard_02"]

train_maps3 = ["easy_01",
               "medium_01", "medium_02", "medium_03", "medium_04",
              "hard_01", "hard_02", "hard_03", "hard_04"]

test_maps  = ["easy_02",
              "medium_05", "medium_06",
              "hard_05", "hard_06"]


def make_env(train_layouts, test_layouts, split, seed=0):
    env = gym.make(
        'gymnasium_env/PacmanGen-v0',
        seed=seed, 
        render_or_not=False, 
        render_mode="rgb_array",
        train_layouts=train_layouts, 
        test_layouts=test_layouts, 
        split=split,
        max_steps=300
    )
    return env

env = make_env(train_maps1, test_maps, split="train")
# env = make_env(train_maps2, test_maps, split="train")
# env = make_env(train_maps3, test_maps, split="train")

class ApproxQLearningPacman:
    def __init__(
        self,
        extractor,
        alpha=0.05,
        gamma=0.99,
        epsilon=0.10,
        epsilon_decay=None,
        epsilon_min=0.01,
    ):
        """
        Approximate Q-learning using sparse feature weights.

        Args:
            extractor: AdvancedExtractor()
            alpha: learning rate
            gamma: discount factor
            epsilon: exploration rate
            epsilon_decay: optional multiplicative decay per episode, e.g. 0.995
            epsilon_min: lower bound for epsilon if decay is used
        """
        self.extractor = extractor
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # sparse weights: feature_name -> scalar
        self.weights = defaultdict(float)

    def getQValue(self, state, action_str):
        """
        Q(s,a) = sum_f w_f * phi_f(s,a)
        state: Pacman game state
        action_str: one of "Stop", "North", "South", "West", "East"
        """
        qValue = 0.0
        features = self.extractor.getFeatures(state, action_str)
        
        for key in list(features.keys()):
            qValue = qValue + self.weights[key] * features[key]
        return qValue


    def getValue(self, env, state):
        valuesForActions = Counter()
        legal_actions = state.getLegalActions(0)

        if not legal_actions:
            return 0.0
        
        for a in legal_actions:
            valuesForActions[a] = self.getQValue(state, a)
        max_a_value = valuesForActions[valuesForActions.argMax()]
        return max_a_value

    def getPolicy(self, env, state):
        valuesForActions = Counter()

        legal_actions = state.getLegalActions(0)

        if not legal_actions:
            return None
    
        for a in legal_actions:
            valuesForActions[a] = self.getQValue(state, a)
        best_action = valuesForActions.argMax()

        return best_action

    def getAction(self, env, state):
        legal_actions = state.getLegalActions(0)
        bestAction = self.getPolicy(env, state)

        if not legal_actions:
            return None

        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        
        return bestAction

    def update(self, env, state, action_str, next_state, reward, done):
        """
        Weight update:
        difference = [r + gamma * max_a' Q(s',a')] - Q(s,a)
        w_f <- w_f + alpha * difference * feature_f
        """
        features = self.extractor.getFeatures(state, action_str)
        q_sa = self.getQValue(state, action_str)

        if done:
            target = reward
        else:
            target = reward + self.gamma * self.getValue(env, next_state)

        td_error = target - q_sa

        for f, value in features.items():
            self.weights[f] += self.alpha * td_error * value

    def decay_epsilon(self):
        if self.epsilon_decay is not None:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train_approx_q_agent(
    env,
    agent,
    num_episodes=500,
    evaluate_every=25,
    eval_episodes=10,
):
    """
    Trains approximate Q-learning directly on PacmanEnv.

    Important:
    - env.reset() returns an observation, but the feature extractor uses env.game.state
    - env.step(action_idx) expects integer actions
    """
    episode_returns = []
    eval_returns = []

    # mapping between env integer actions and Pacman action strings
    idx_to_action = env.get_action_meanings()  # ["Stop", "North", "South", "West", "East"]
    action_to_idx = {a: i for i, a in enumerate(idx_to_action)}

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            state = env.game.state  # use true Pacman state for features

            action_str = agent.getAction(env, state)
            if action_str is None:
                break

            action_idx = action_to_idx[action_str]

            obs_next, reward, terminated, truncated, info = env.step(action_idx)
            done = terminated or truncated
            next_state = env.game.state

            agent.update(env, state, action_str, next_state, reward, done)

            total_reward += reward

        episode_returns.append(total_reward)
        agent.decay_epsilon()

        if episode % evaluate_every == 0:
            avg_eval = evaluate_approx_q_agent(env, agent, num_episodes=eval_episodes)
            eval_returns.append(avg_eval)
            print(
                f"Episode {episode:4d} | "
                f"train return: {total_reward:8.2f} | "
                f"eval avg: {avg_eval:8.2f} | "
                f"epsilon: {agent.epsilon:.4f}"
            )

    return episode_returns, eval_returns

def evaluate_approx_q_agent(env, agent, num_episodes=10):
    """
    Greedy evaluation.
    """
    returns = []

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        idx_to_action = env.get_action_meanings()
        action_to_idx = {a: i for i, a in enumerate(idx_to_action)}

        while not done:
            state = env.game.state
            action_str = agent.getPolicy(env, state)

            if action_str is None:
                break

            action_idx = action_to_idx[action_str]
            obs, reward, terminated, truncated, info = env.step(action_idx)

            done = terminated or truncated
            total_reward += reward

        returns.append(total_reward)

    agent.epsilon = old_epsilon
    return float(np.mean(returns))


extractor = AdvancedExtractor()

agent = ApproxQLearningPacman(
    extractor=extractor,
    alpha=0.05,
    gamma=0.99,
    epsilon=0.10,
    epsilon_decay=None,
    epsilon_min=None,
)

train_returns, eval_returns = train_approx_q_agent(
    env,
    agent,
    num_episodes=500,
    evaluate_every=25,
    eval_episodes=10,
)

for k, v in sorted(agent.weights.items(), key=lambda x: -abs(x[1])):
    print(f"{k:35s} {v: .4f}")
