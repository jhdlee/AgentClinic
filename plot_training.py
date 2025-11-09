#!/usr/bin/env python3
"""
Plot training progress from epoch statistics.

Usage:
    python plot_training.py --stats_file path/to/epoch_stats.json
    python plot_training.py --output_dir path/to/outputs/ppo_run
"""

import argparse
import json
import os
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np


def load_epoch_stats(stats_file: str) -> List[Dict[str, Any]]:
    """Load epoch statistics from JSON file."""
    with open(stats_file, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_training_curves(epoch_stats: List[Dict[str, Any]], output_path: str = None):
    """
    Plot training curves for reward, accuracy, and interactions.

    Args:
        epoch_stats: List of dictionaries with epoch statistics
        output_path: Optional path to save the figure
    """
    epochs = [stat["epoch"] for stat in epoch_stats]
    avg_rewards = [stat["avg_reward"] for stat in epoch_stats]
    avg_accuracy = [stat["avg_accuracy"] for stat in epoch_stats]
    avg_interactions = [stat["avg_interactions"] for stat in epoch_stats]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot average reward
    axes[0].plot(epochs, avg_rewards, marker='o', linewidth=2, markersize=6)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Average Reward", fontsize=12)
    axes[0].set_title("Training Reward Progress", fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Plot average accuracy
    axes[1].plot(epochs, avg_accuracy, marker='s', linewidth=2, markersize=6, color='green')
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Average Accuracy", fontsize=12)
    axes[1].set_title("Training Accuracy Progress", fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 1.05])
    axes[1].grid(True, alpha=0.3)

    # Plot average interactions
    axes[2].plot(epochs, avg_interactions, marker='^', linewidth=2, markersize=6, color='orange')
    axes[2].set_xlabel("Epoch", fontsize=12)
    axes[2].set_ylabel("Average Interactions", fontsize=12)
    axes[2].set_title("Average Interactions per Episode", fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def print_summary(epoch_stats: List[Dict[str, Any]]):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Total epochs: {len(epoch_stats)}")
    print(f"\nFirst epoch:")
    print(f"  Reward: {epoch_stats[0]['avg_reward']:.3f}")
    print(f"  Accuracy: {epoch_stats[0]['avg_accuracy']:.3f}")
    print(f"  Avg interactions: {epoch_stats[0]['avg_interactions']:.2f}")
    print(f"\nFinal epoch:")
    print(f"  Reward: {epoch_stats[-1]['avg_reward']:.3f}")
    print(f"  Accuracy: {epoch_stats[-1]['avg_accuracy']:.3f}")
    print(f"  Avg interactions: {epoch_stats[-1]['avg_interactions']:.2f}")
    print(f"\nImprovement:")
    reward_improvement = epoch_stats[-1]['avg_reward'] - epoch_stats[0]['avg_reward']
    accuracy_improvement = epoch_stats[-1]['avg_accuracy'] - epoch_stats[0]['avg_accuracy']
    print(f"  Reward: {reward_improvement:+.3f}")
    print(f"  Accuracy: {accuracy_improvement:+.3f}")
    print(f"  Avg interactions: {epoch_stats[-1]['avg_interactions'] - epoch_stats[0]['avg_interactions']:+.2f}")

    # Find best epoch
    best_accuracy_idx = np.argmax([s['avg_accuracy'] for s in epoch_stats])
    best_reward_idx = np.argmax([s['avg_reward'] for s in epoch_stats])
    print(f"\nBest accuracy: {epoch_stats[best_accuracy_idx]['avg_accuracy']:.3f} (epoch {epoch_stats[best_accuracy_idx]['epoch']})")
    print(f"Best reward: {epoch_stats[best_reward_idx]['avg_reward']:.3f} (epoch {epoch_stats[best_reward_idx]['epoch']})")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Plot PPO training progress")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stats_file", type=str, help="Path to epoch_stats.json file")
    group.add_argument("--output_dir", type=str, help="Path to output directory containing epoch_stats.json")
    parser.add_argument("--save_plot", type=str, default=None, help="Path to save the plot (default: show interactive plot)")
    parser.add_argument("--no_summary", action="store_true", help="Skip printing summary statistics")
    args = parser.parse_args()

    # Determine stats file path
    if args.output_dir:
        stats_file = os.path.join(args.output_dir, "epoch_stats.json")
    else:
        stats_file = args.stats_file

    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"Statistics file not found: {stats_file}")

    # Load epoch statistics
    print(f"Loading statistics from {stats_file}...")
    epoch_stats = load_epoch_stats(stats_file)

    if not epoch_stats:
        print("No epoch statistics found in file.")
        return

    # Print summary
    if not args.no_summary:
        print_summary(epoch_stats)

    # Plot training curves
    output_path = args.save_plot
    if output_path is None and args.output_dir:
        output_path = os.path.join(args.output_dir, "training_curves.png")

    plot_training_curves(epoch_stats, output_path)


if __name__ == "__main__":
    main()
