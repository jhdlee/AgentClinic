#!/usr/bin/env python3
"""
Plot training metrics from AgentClinic PPO training outputs.

This script parses the episode txt files in the llm_outputs directory and creates
plots showing how average reward, correctness, and number of interactions change
over training epochs.

Usage:
    python plot_training_metrics.py --output_dir /path/to/output_dir
    python plot_training_metrics.py --output_dir /scratch/users/hdlee/agent_clinic/outputs/ppo_budget_aware
"""

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_episode_file(filepath: Path) -> Dict[str, any]:
    """
    Parse a single episode output txt file to extract metrics.

    Returns:
        Dict with keys: epoch, episode_num, scenario_id, num_turns, correctness, total_reward, budget_saved
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    metrics = {}

    # Extract epoch (handle both numeric and eval formats)
    epoch_match = re.search(r'Epoch:\s+(\d+)', content)
    if epoch_match:
        metrics['epoch'] = int(epoch_match.group(1))
    elif 'TRAIN EVALUATION' in content:
        metrics['epoch'] = 'train_eval'
    elif 'TEST EVALUATION' in content:
        metrics['epoch'] = 'test_eval'
    else:
        return None  # Skip if epoch not found

    # Extract episode number
    episode_match = re.search(r'Episode Number:\s+(\d+)', content)
    if episode_match:
        metrics['episode_num'] = int(episode_match.group(1))

    # Extract scenario ID
    scenario_match = re.search(r'Scenario ID:\s+(\d+)', content)
    if scenario_match:
        metrics['scenario_id'] = int(scenario_match.group(1))

    # Extract number of turns
    turns_match = re.search(r'Number of Turns:\s+(\d+)', content)
    if turns_match:
        metrics['num_turns'] = int(turns_match.group(1))

    # Extract correctness
    correctness_match = re.search(r'Correctness:\s+([-+]?\d*\.?\d+)', content)
    if correctness_match:
        metrics['correctness'] = float(correctness_match.group(1))

    # Extract total reward
    reward_match = re.search(r'Total Reward:\s+([-+]?\d*\.?\d+)', content)
    if reward_match:
        metrics['total_reward'] = float(reward_match.group(1))

    # Extract budget saved
    budget_match = re.search(r'Budget Saved:\s+([-+]?\d*\.?\d+)', content)
    if budget_match:
        metrics['budget_saved'] = float(budget_match.group(1))

    return metrics


def parse_all_episodes(llm_outputs_dir: Path) -> List[Dict[str, any]]:
    """
    Parse all episode txt files in the llm_outputs directory.

    Returns:
        List of dicts, each containing metrics for one episode
    """
    all_metrics = []

    txt_files = sorted(llm_outputs_dir.glob("epoch_*.txt"))

    if not txt_files:
        print(f"Warning: No episode files found in {llm_outputs_dir}")
        return []

    print(f"Found {len(txt_files)} episode files")

    for filepath in txt_files:
        metrics = parse_episode_file(filepath)
        if metrics and isinstance(metrics['epoch'], int):  # Only include training epochs
            all_metrics.append(metrics)

    print(f"Parsed {len(all_metrics)} training episodes")
    return all_metrics


def compute_epoch_statistics(all_metrics: List[Dict[str, any]]) -> Dict[int, Dict[str, float]]:
    """
    Group episodes by epoch and compute average metrics.

    For correctness, treats -1 as 0 (incorrect) for averaging purposes.

    Returns:
        Dict mapping epoch number to dict of average metrics
    """
    epoch_data = defaultdict(lambda: {'rewards': [], 'correctness': [], 'num_turns': []})

    for metrics in all_metrics:
        epoch = metrics['epoch']

        if 'total_reward' in metrics:
            epoch_data[epoch]['rewards'].append(metrics['total_reward'])

        if 'correctness' in metrics:
            # Convert -1 to 0 for averaging (incorrect = 0, correct = 1)
            correctness_normalized = max(0.0, metrics['correctness'])
            epoch_data[epoch]['correctness'].append(correctness_normalized)

        if 'num_turns' in metrics:
            epoch_data[epoch]['num_turns'].append(metrics['num_turns'])

    # Compute averages
    epoch_stats = {}
    for epoch, data in epoch_data.items():
        epoch_stats[epoch] = {
            'avg_reward': np.mean(data['rewards']) if data['rewards'] else 0,
            'avg_correctness': np.mean(data['correctness']) if data['correctness'] else 0,
            'avg_num_turns': np.mean(data['num_turns']) if data['num_turns'] else 0,
            'std_reward': np.std(data['rewards']) if data['rewards'] else 0,
            'std_correctness': np.std(data['correctness']) if data['correctness'] else 0,
            'std_num_turns': np.std(data['num_turns']) if data['num_turns'] else 0,
            'num_episodes': len(data['rewards']),
        }

    return epoch_stats


def plot_training_metrics(epoch_stats: Dict[int, Dict[str, float]], output_dir: Path):
    """
    Create plots showing how metrics change over epochs.
    """
    # Sort epochs
    epochs = sorted(epoch_stats.keys())

    if not epochs:
        print("No data to plot!")
        return

    # Extract data for plotting
    avg_rewards = [epoch_stats[e]['avg_reward'] for e in epochs]
    avg_correctness = [epoch_stats[e]['avg_correctness'] for e in epochs]
    avg_num_turns = [epoch_stats[e]['avg_num_turns'] for e in epochs]

    std_rewards = [epoch_stats[e]['std_reward'] for e in epochs]
    std_correctness = [epoch_stats[e]['std_correctness'] for e in epochs]
    std_num_turns = [epoch_stats[e]['std_num_turns'] for e in epochs]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Plot 1: Average Reward
    axes[0].plot(epochs, avg_rewards, 'o-', linewidth=2, markersize=6, color='#2E86AB', label='Avg Reward')
    axes[0].fill_between(
        epochs,
        np.array(avg_rewards) - np.array(std_rewards),
        np.array(avg_rewards) + np.array(std_rewards),
        alpha=0.2,
        color='#2E86AB'
    )
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Average Reward', fontsize=12)
    axes[0].set_title('Average Reward per Epoch', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot 2: Average Correctness
    axes[1].plot(epochs, avg_correctness, 'o-', linewidth=2, markersize=6, color='#A23B72', label='Avg Correctness')
    axes[1].fill_between(
        epochs,
        np.array(avg_correctness) - np.array(std_correctness),
        np.array(avg_correctness) + np.array(std_correctness),
        alpha=0.2,
        color='#A23B72'
    )
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Average Correctness', fontsize=12)
    axes[1].set_title('Average Correctness per Epoch (Accuracy)', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 1.1])  # Correctness is between 0 and 1
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Plot 3: Average Number of Turns
    axes[2].plot(epochs, avg_num_turns, 'o-', linewidth=2, markersize=6, color='#F18F01', label='Avg Num Turns')
    axes[2].fill_between(
        epochs,
        np.array(avg_num_turns) - np.array(std_num_turns),
        np.array(avg_num_turns) + np.array(std_num_turns),
        alpha=0.2,
        color='#F18F01'
    )
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Average Number of Turns', fontsize=12)
    axes[2].set_title('Average Number of Interactions per Epoch', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()

    # Save figure
    plot_path = output_dir / 'training_metrics.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to: {plot_path}")

    # Also save as PDF for publication quality
    plot_path_pdf = output_dir / 'training_metrics.pdf'
    plt.savefig(plot_path_pdf, bbox_inches='tight')
    print(f"Saved plot to: {plot_path_pdf}")

    plt.show()

    # Print summary statistics
    print("\n" + "="*60)
    print("TRAINING SUMMARY STATISTICS")
    print("="*60)
    for epoch in epochs:
        stats = epoch_stats[epoch]
        print(f"\nEpoch {epoch} (n={stats['num_episodes']} episodes):")
        print(f"  Avg Reward:      {stats['avg_reward']:.3f} ± {stats['std_reward']:.3f}")
        print(f"  Avg Correctness: {stats['avg_correctness']:.3f} ± {stats['std_correctness']:.3f}")
        print(f"  Avg Num Turns:   {stats['avg_num_turns']:.2f} ± {stats['std_num_turns']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot training metrics from AgentClinic PPO outputs"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Path to the output directory containing llm_outputs folder'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    llm_outputs_dir = output_dir / 'llm_outputs'

    if not llm_outputs_dir.exists():
        print(f"Error: llm_outputs directory not found at {llm_outputs_dir}")
        print(f"Make sure you've run training with --save_llm_outputs flag")
        return

    print(f"Reading episode files from: {llm_outputs_dir}")

    # Parse all episode files
    all_metrics = parse_all_episodes(llm_outputs_dir)

    if not all_metrics:
        print("No training episodes found to plot!")
        return

    # Compute epoch-level statistics
    epoch_stats = compute_epoch_statistics(all_metrics)

    # Create plots
    plot_training_metrics(epoch_stats, output_dir)


if __name__ == '__main__':
    main()
