# Approximate Q-learning implementation for PacmanGen environment
import gc
gc.collect()

import gymnasium as gym
import numpy as np
import random
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

from feature_extractors import AdvancedExtractor
from pacman_gym.envs.pacman.util import Counter

import itertools
import time
import pickle

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
    return env.unwrapped

env1 = make_env(train_maps1, test_maps, split="train")
test_env1 = make_env(train_maps1, train_maps1+test_maps, split="test")

env2 = make_env(train_maps2, test_maps, split="train")
test_env2 = make_env(train_maps2, train_maps2+test_maps, split="test")

env3 = make_env(train_maps3, test_maps, split="train")
test_env3 = make_env(train_maps3, train_maps3+test_maps, split="test")

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

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(dict(self.weights), f)

    def load(self, path):
        with open(path, "rb") as f:
            self.weights = defaultdict(float, pickle.load(f))

def train_approx_q_agent(
    env,
    agent,
    num_episodes=1000,
    evaluate_every=25,
    eval_episodes=10,
    log_dir="runs/approx_qlearning",
    start_episode=0,
):
    """
    Trains approximate Q-learning directly on PacmanEnv.
    """
    writer = SummaryWriter(log_dir=log_dir, purge_step=start_episode)
    episode_returns = []
    eval_returns = []
    eval_win_rates = []

    # mapping between env integer actions and Pacman action strings
    idx_to_action = env.get_action_meanings()  # ["Stop", "North", "South", "West", "East"]
    action_to_idx = {a: i for i, a in enumerate(idx_to_action)}

    for episode in range(1+start_episode, start_episode+num_episodes + 1):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        episode_steps = 0

        while not done:
            state = env.game.state

            action_str = agent.getAction(env, state)
            if action_str is None:
                break

            action_idx = action_to_idx[action_str]

            obs_next, reward, terminated, truncated, info = env.step(action_idx)    
            done = terminated or truncated
            
            next_state = env.game.state

            agent.update(env, state, action_str, next_state, reward, done)

            total_reward += reward
            episode_steps += 1

        episode_returns.append(total_reward)
        agent.decay_epsilon()

  
        if episode % evaluate_every == 0:
            writer.add_scalar("train/episode_return", total_reward, episode)
            writer.add_scalar("train/epsilon", agent.epsilon, episode)
            writer.add_scalar("train/episode_steps", episode_steps, episode)

            if len(episode_returns) >= evaluate_every:
                writer.add_scalar(
                    f"train/return_mean_{evaluate_every}",
                    np.mean(episode_returns[-evaluate_every:]),
                    episode,
                )

            eval_stats = evaluate_approx_q_agent(env, agent, num_episodes=eval_episodes)

            eval_returns.append(eval_stats["avg_return"])
            eval_win_rates.append(eval_stats["win_rate"])

            writer.add_scalar("eval/avg_return", eval_stats["avg_return"], episode)
            writer.add_scalar("eval/win_rate", eval_stats["win_rate"], episode)
            writer.add_scalar("eval/avg_final_score", eval_stats["avg_final_score"], episode)
            writer.add_scalar("eval/avg_percent_food_eaten", eval_stats["avg_percent_food_eaten"], episode)
            writer.add_scalar("eval/avg_normalized_score", eval_stats["avg_normalized_score"], episode)

                # Weight stats
            if len(agent.weights) > 0:
                weight_values = np.array(list(agent.weights.values()), dtype=np.float32)
                writer.add_scalar("weights/l2_norm", np.linalg.norm(weight_values), episode)
                writer.add_scalar("weights/max_abs", np.max(np.abs(weight_values)), episode)
                writer.add_scalar("weights/num_nonzero", np.count_nonzero(weight_values), episode)


            print(
                f"Episode {episode:4d} | "
                f"train return: {total_reward:8.2f} | "
                f"eval avg: {eval_stats['avg_return']:8.2f} | "
                f"win rate: {eval_stats['win_rate']:.2f} | "
                f"epsilon: {agent.epsilon:.4f}"
            )

    return episode_returns, eval_returns, eval_win_rates

def evaluate_approx_q_agent(env, agent, num_episodes=500):
    """
    Greedy evaluation.
    """
    returns = []
    successes = []
    final_scores = []
    percent_food_eaten = []
    normalized_scores = []

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    idx_to_action = env.get_action_meanings()
    action_to_idx = {a: i for i, a in enumerate(idx_to_action)}


    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        last_info = {}


        while not done:
            state = env.game.state
            action_str = agent.getPolicy(env, state)

            if action_str is None:
                break

            action_idx = action_to_idx[action_str]
            obs, reward, terminated, truncated, info = env.step(action_idx)

            done = terminated or truncated
            total_reward += reward
            last_info = info

        returns.append(total_reward)
        successes.append(float(last_info.get("is_success", 0.0)))
        final_scores.append(float(last_info.get("final_score", 0.0)))
        percent_food_eaten.append(float(last_info.get("percent_food_eaten", 0.0)))
        normalized_scores.append(float(last_info.get("normalized_score", 0.0)))


    agent.epsilon = old_epsilon
    return {
        "avg_return": float(np.mean(returns)),
        "win_rate": float(np.mean(successes)),
        "avg_final_score": float(np.mean(final_scores)),
        "avg_percent_food_eaten": float(np.mean(percent_food_eaten)),
        "avg_normalized_score": float(np.mean(normalized_scores)),
    }

###############################################
# Hyperparameter tuning
###############################################

# param_grid = {
#     "alpha": [0.05, 0.1],
#     "gamma": [0.95, 0.99],
#     "epsilon": [0.1, 0.2],
#     "epsilon_decay": [0.99, 0.995],
# }

# results = []

# for alpha, gamma, epsilon, epsilon_decay in itertools.product(
#     param_grid["alpha"],
#     param_grid["gamma"],
#     param_grid["epsilon"],
#     param_grid["epsilon_decay"],
# ):
#     print("\n==============================")
#     print(f"alpha={alpha}, gamma={gamma}, eps={epsilon}, decay={epsilon_decay}")

#     # new extractor
#     extractor = AdvancedExtractor()

#     # new agent 
#     agent = ApproxQLearningPacman(
#         extractor=extractor,
#         alpha=alpha,
#         gamma=gamma,
#         epsilon=epsilon,
#         epsilon_decay=epsilon_decay,
#         epsilon_min=0.01,
#     )

#     log_dir = f"runs/grid/alpha{alpha}_gamma{gamma}_eps{epsilon}_decay{epsilon_decay}_{int(time.time())}"

#     episode_returns, eval_returns, eval_win_rates = train_approx_q_agent(
#         env1,
#         agent,
#         num_episodes=500,
#         evaluate_every=100,
#         eval_episodes=50,
#         log_dir=log_dir,
#     )

#     # do one final evaluation explicitly
#     final_stats = evaluate_approx_q_agent(env2, agent, num_episodes=50)

#     results.append({
#         "alpha": alpha,
#         "epsilon": epsilon,
#         "final_eval": final_stats["avg_return"],
#         "win_rate": final_stats["win_rate"],
#         "avg_final_score": final_stats["avg_final_score"],
#         "avg_percent_food_eaten": final_stats["avg_percent_food_eaten"],
#         "avg_normalized_score": final_stats["avg_normalized_score"],
#     })

# best = max(results, key=lambda x: (x["win_rate"], x["final_eval"]))

# print("\nBEST CONFIG:")
# for k, v in best.items():
#     print(f"{k}: {v}")

# results_sorted = sorted(results, key=lambda x: (x["win_rate"], x["final_eval"]), reverse=True)
# for r in results_sorted:
#     print(r)


#########################################################
# Training best hyperparameter combinations
#########################################################

extractor = AdvancedExtractor()

agent = ApproxQLearningPacman(
    extractor=extractor,
    alpha=0.05,
    gamma=0.99,
    epsilon=0.2,
    epsilon_decay=0.997,
    epsilon_min=0.01,
)

train_returns, eval_returns, eval_win_rates = train_approx_q_agent(
    env2,
    agent,
    num_episodes=1000,
    evaluate_every=200,
    eval_episodes=100,
    log_dir="runs/env2_alpha0.05_gamma0.99_eps0.2_decay0.997",
    start_episode=0,
    )

for k, v in sorted(agent.weights.items(), key=lambda x: -abs(x[1])):
    print(f"{k:35s} {v: .4f}")

agent.save("aql_env2_new.pkl")

# best one for env1 is hyp1 with alpha=0.1, gamma=0.95, epsilon=0.2, decay=0.995

#########################################################
# Training best chosen hyperparameters on env1
#########################################################

import os
os.makedirs("results", exist_ok=True)

#del agent
#gc.collect()

extractor = AdvancedExtractor()

agent = ApproxQLearningPacman(
    extractor=extractor,
    alpha=0.1,
    gamma=0.95,
    epsilon=0.2, 
    epsilon_decay=0.995,
    epsilon_min=0.01,
)

# agent.load("aql_env1.pkl")

train_returns, eval_returns, eval_win_rates = train_approx_q_agent(
    env1,
    agent,
    num_episodes=500,
    evaluate_every=100,
    eval_episodes=100,
    log_dir="runs/approx_qlearning_env1",
    start_episode=0,
    )

agent.save("aql_env1.pkl")

for k, v in sorted(agent.weights.items(), key=lambda x: -abs(x[1])):
    print(f"{k:35s} {v: .4f}")

np.savez(
    "results/env1_results.npz",
    train_returns=train_returns,
    eval_returns=eval_returns,
    eval_win_rates=eval_win_rates,
    )

#########################################################
# Training best chosen hyperparameters on env2
#########################################################

del agent
gc.collect()

extractor = AdvancedExtractor()

agent = ApproxQLearningPacman(
    extractor=extractor,
    alpha=0.1,
    gamma=0.95,
    epsilon=0.2, 
    epsilon_decay=0.995,
    epsilon_min=0.01,
)

# agent.load("aql_env2.pkl")

train_returns, eval_returns, eval_win_rates = train_approx_q_agent(
    env2,
    agent,
    num_episodes=1500,
    evaluate_every=100,
    eval_episodes=100,
    log_dir="runs/approx_qlearning_env2",
    start_episode=0,
    )

agent.save("aql_env2.pkl")

for k, v in sorted(agent.weights.items(), key=lambda x: -abs(x[1])):
    print(f"{k:35s} {v: .4f}")

np.savez(
    "results/env2_results.npz",
    train_returns=train_returns,
    eval_returns=eval_returns,
    eval_win_rates=eval_win_rates,
    )

#########################################################
# Training best chosen hyperparameters on env3
#########################################################

del agent
gc.collect()

extractor = AdvancedExtractor()

agent = ApproxQLearningPacman(
    extractor=extractor,
    alpha=0.1,
    gamma=0.95,
    epsilon=0.2, 
    epsilon_decay=0.995,
    epsilon_min=0.01,
)

# agent.load("aql_env3.pkl")

train_returns, eval_returns, eval_win_rates = train_approx_q_agent(
    env3,
    agent,
    num_episodes=1500,
    evaluate_every=100,
    eval_episodes=100,
    log_dir="runs/approx_qlearning_env3",
    start_episode=0,
    )

agent.save("aql_env3.pkl")

for k, v in sorted(agent.weights.items(), key=lambda x: -abs(x[1])):
    print(f"{k:35s} {v: .4f}")

np.savez(
    "results/env3_results.npz",
    train_returns=train_returns,
    eval_returns=eval_returns,
    eval_win_rates=eval_win_rates,
    )