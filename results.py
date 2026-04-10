# Functions to generate results and summaries

from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

def evaluate_dqn_by_layout(model, env, n_eval_episodes=500, print_results=True):
    results_by_layout = defaultdict(list)

    for episode in range(n_eval_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        print(f"Evaluating episode {episode+1}/{n_eval_episodes}...", end="\r")

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
    
    summary_by_layout = pd.DataFrame(summary_by_layout).T
    summary_by_layout = summary_by_layout.reset_index().rename(columns={"index": "layout_name"})

    return results_by_layout, summary_by_layout


def evaluate_approx_q_by_layout(agent, env, n_eval_episodes=500, print_results=True):
    results_by_layout = defaultdict(list)

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0   # greedy evaluation

    idx_to_action = env.get_action_meanings()
    action_to_idx = {a: i for i, a in enumerate(idx_to_action)}

    for episode in range(n_eval_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        print(f"Evaluating episode {episode+1}/{n_eval_episodes}...", end="\r")

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
    
    summary_by_layout = pd.DataFrame(summary_by_layout).T
    summary_by_layout = summary_by_layout.reset_index().rename(columns={"index": "layout_name"})

    return results_by_layout, summary_by_layout


def get_pastel_colors(df, test_maps, train_maps):
    train_color = "#52A9AC"
    test_color = "#C66161"

    return [
        test_color if layout in test_maps else train_color
        for layout in df["layout_name"]
    ]


def plot_layout_summary(df, env_name=None, 
                        train_maps=None, test_maps=None,
                        save_path=None, figsize=(10, 2)): 
    
    if test_maps is None:
        test_maps = []
    if train_maps is None:
        train_maps = []

    # assign colors
    df["color"] = get_pastel_colors(df, test_maps, train_maps)

    # optional sort
    df = df.sort_values(["color"])

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].barh(df["layout_name"], df["mean_reward"], color=df["color"])
    axes[0].set_title("Mean Reward")

    axes[1].barh(df["layout_name"], df["mean_percent_food_eaten"], color=df["color"])
    axes[1].set_title("Food Completion Rate")
    axes[1].set_xlim(0, 1.05)
    axes[1].set_yticks([])

    legend_elements = [
        Patch(facecolor="#52A9AC", label=f"Train Layouts {env_name}"),
        Patch(facecolor="#C66161", label="Test Layouts"),
    ]

    axes[1].legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False
    )

    for ax in axes:
        ax.invert_yaxis()
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.4)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()