# Functions to generate results and summaries

from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def evaluate_dqn_by_layout(model, env, n_eval_episodes=700, print_results=True):
    results_by_layout = defaultdict(list)

    for episode in range(n_eval_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        layout_name = info["layout_name"]

        results_by_layout[layout_name].append({
            "episode_reward": episode_reward,
            "is_success": info.get("is_success", False),
            "final_score": info.get("final_score", 0.0),
            "maxsteps_used": info.get("maxsteps_used", False),
            "initial_num_food": info.get("initial_num_food", 0),
            "remaining_food": info.get("remaining_food", 0),
            "percent_food_eaten": info.get("percent_food_eaten", 0.0),
            "normalized_score": info.get("normalized_score", 0.0),
        })

    env.close()

    summary_by_layout = {}

    for layout_name, episodes in results_by_layout.items():
        rewards = [ep["episode_reward"] for ep in episodes]
        wins = [ep["is_success"] for ep in episodes]
        final_scores = [ep["final_score"] for ep in episodes]
        food_eaten = [ep["percent_food_eaten"] for ep in episodes]
        normalized_scores = [ep["normalized_score"] for ep in episodes]
        maxsteps_used = [ep["maxsteps_used"] for ep in episodes]

        summary_by_layout[layout_name] = {
            "num_episodes": len(episodes),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "win_rate": float(np.mean(wins)),
            "mean_final_score": float(np.mean(final_scores)),
            "mean_percent_food_eaten": float(np.mean(food_eaten)),
            "mean_normalized_score": float(np.mean(normalized_scores)),
            "maxsteps_rate": float(np.mean(maxsteps_used))
            }

    if print_results:
        for layout_name, stats in summary_by_layout.items():
            print(f"\nLayout: {layout_name}")
            for k, v in stats.items():
                print(f"  {k}: {v}")
    

    return results_by_layout, summary_by_layout


def evaluate_approx_q_by_layout(agent, env, n_eval_episodes=700, print_results=True):
    results_by_layout = defaultdict(list)

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0   # greedy evaluation

    idx_to_action = env.get_action_meanings()
    action_to_idx = {a: i for i, a in enumerate(idx_to_action)}

    for episode in range(n_eval_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            state = env.game.state
            action_str = agent.getPolicy(env, state)

            if action_str is None:
                break

            action_idx = action_to_idx[action_str]
            obs, reward, terminated, truncated, info = env.step(action_idx)

            episode_reward += reward
            done = terminated or truncated

        layout_name = info["layout_name"]

        results_by_layout[layout_name].append({
            "episode_reward": episode_reward,
            "is_success": info.get("is_success", False),
            "final_score": info.get("final_score", 0.0),
            "maxsteps_used": info.get("maxsteps_used", False),
            "initial_num_food": info.get("initial_num_food", 0),
            "remaining_food": info.get("remaining_food", 0),
            "percent_food_eaten": info.get("percent_food_eaten", 0.0),
            "normalized_score": info.get("normalized_score", 0.0),
        })

    agent.epsilon = old_epsilon
    env.close()

    summary_by_layout = {}

    for layout_name, episodes in results_by_layout.items():
        rewards = [ep["episode_reward"] for ep in episodes]
        wins = [ep["is_success"] for ep in episodes]
        final_scores = [ep["final_score"] for ep in episodes]
        food_eaten = [ep["percent_food_eaten"] for ep in episodes]
        normalized_scores = [ep["normalized_score"] for ep in episodes]
        maxsteps_used = [ep["maxsteps_used"] for ep in episodes]

        summary_by_layout[layout_name] = {
            "num_episodes": len(episodes),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "win_rate": float(np.mean(wins)),
            "mean_final_score": float(np.mean(final_scores)),
            "mean_percent_food_eaten": float(np.mean(food_eaten)),
            "mean_normalized_score": float(np.mean(normalized_scores)),
            "maxsteps_rate": float(np.mean(maxsteps_used)),
        }

    if print_results:
        for layout_name, stats in summary_by_layout.items():
            print(f"\nLayout: {layout_name}")
            for k, v in stats.items():
                print(f"  {k}: {v}")

    return results_by_layout, summary_by_layout


def plot_layout_summary(summary_by_layout):
    df = pd.DataFrame(summary_by_layout).T
    df = df.reset_index().rename(columns={"index": "layout_name"})

    # optional: sort layouts by name
    df = df.sort_values("layout_name")

    # 1. Mean reward
    plt.figure(figsize=(8, 4))
    plt.bar(df["layout_name"], df["mean_reward"])
    plt.title("Mean Reward by Layout")
    plt.xlabel("Layout")
    plt.ylabel("Mean Reward")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 2. Win rate
    plt.figure(figsize=(8, 4))
    plt.bar(df["layout_name"], df["win_rate"])
    plt.title("Win Rate by Layout")
    plt.xlabel("Layout")
    plt.ylabel("Win Rate")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 3. Mean percent food eaten
    plt.figure(figsize=(8, 4))
    plt.bar(df["layout_name"], df["mean_percent_food_eaten"])
    plt.title("Mean Percent Food Eaten by Layout")
    plt.xlabel("Layout")
    plt.ylabel("Percent Food Eaten")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 4. Mean normalized score
    plt.figure(figsize=(8, 4))
    plt.bar(df["layout_name"], df["mean_normalized_score"])
    plt.title("Mean Normalized Score by Layout")
    plt.xlabel("Layout")
    plt.ylabel("Normalized Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return df