#!/usr/bin/env python3
"""
Plot results from max_turns sweep.

This script loads sweep results and creates plots showing:
1. Max turns (limit) vs test accuracy
2. Average turns taken vs test accuracy

Usage:
    python plot_sweep.py --results_dir outputs/sweep_max_turns
    python plot_sweep.py --results_dir outputs/sweep_max_turns --save_plot sweep_results.png
"""

import argparse
import json
import os
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_sweep_results(results_dir: str) -> Dict[str, Any]:
    """Load sweep results from JSON file."""
    results_path = os.path.join(results_dir, "sweep_results.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Sweep results not found at {results_path}")

    with open(results_path, "r") as f:
        return json.load(f)


def extract_metrics(
    sweep_results: Dict[str, Any],
    mode: str
) -> Tuple[List[int], List[float], List[float], List[float]]:
    """
    Extract metrics from sweep results.

    Returns:
        max_turns_values: List of max_turns settings
        test_accuracies: List of test accuracies
        avg_turns_taken: List of average turns actually taken
        train_accuracies: List of train accuracies (if available)
    """
    if mode not in sweep_results:
        raise ValueError(f"Mode '{mode}' not found in sweep results")

    results = sweep_results[mode]

    max_turns_values = []
    test_accuracies = []
    avg_turns_taken = []
    train_accuracies = []

    for max_turns_str, data in sorted(results.items(), key=lambda x: int(x[0])):
        max_turns = int(max_turns_str)

        if data.get("status") != "success":
            print(f"Warning: Skipping max_turns={max_turns} (status: {data.get('status')})")
            continue

        # Extract test metrics
        if mode == "train":
            # For training mode, we don't have test eval during training
            # Need to check if evaluation was run after training
            # Look for evaluation_results.json in the output_dir
            output_dir = data.get("output_dir")
            eval_path = os.path.join(output_dir, "evaluation_results.json")
            if os.path.exists(eval_path):
                with open(eval_path, "r") as f:
                    eval_results = json.load(f)
                    if "test" in eval_results:
                        test_accuracies.append(eval_results["test"]["accuracy"])
                        avg_turns_taken.append(eval_results["test"]["avg_interactions"])
                    else:
                        print(f"Warning: No test results for max_turns={max_turns}")
                        continue
                    if "train" in eval_results:
                        train_accuracies.append(eval_results["train"]["accuracy"])
            else:
                # Fallback: use final epoch stats if available
                final_epoch = data.get("final_epoch", {})
                if final_epoch:
                    # For training, we might only have training accuracy
                    train_accuracies.append(final_epoch.get("avg_accuracy", 0.0))
                    avg_turns_taken.append(final_epoch.get("avg_interactions", max_turns))
                    # No test accuracy available
                    test_accuracies.append(None)
                else:
                    continue

        elif mode == "eval":
            eval_results = data.get("eval_results", {})
            if "test" in eval_results:
                test_accuracies.append(eval_results["test"]["accuracy"])
                avg_turns_taken.append(eval_results["test"]["avg_interactions"])
            elif "all" in eval_results:
                test_accuracies.append(eval_results["all"]["accuracy"])
                avg_turns_taken.append(eval_results["all"]["avg_interactions"])
            else:
                print(f"Warning: No test/all results for max_turns={max_turns}")
                continue

            if "train" in eval_results:
                train_accuracies.append(eval_results["train"]["accuracy"])

        max_turns_values.append(max_turns)

    return max_turns_values, test_accuracies, avg_turns_taken, train_accuracies


def plot_sweep_results(
    max_turns_values: List[int],
    test_accuracies: List[float],
    avg_turns_taken: List[float],
    train_accuracies: List[float],
    mode: str,
    output_path: str = None
):
    """
    Create plots showing:
    1. Max turns vs test accuracy
    2. Average turns taken vs test accuracy
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Filter out None values
    valid_indices = [i for i, acc in enumerate(test_accuracies) if acc is not None]
    if not valid_indices:
        print("Error: No valid test accuracies found!")
        return

    max_turns_filtered = [max_turns_values[i] for i in valid_indices]
    test_acc_filtered = [test_accuracies[i] for i in valid_indices]
    avg_turns_filtered = [avg_turns_taken[i] for i in valid_indices]
    train_acc_filtered = [train_accuracies[i] for i in valid_indices] if train_accuracies else None

    # Plot 1: Max turns vs test accuracy
    axes[0].plot(max_turns_filtered, test_acc_filtered, marker='o', linewidth=2,
                 markersize=8, label='Test Accuracy', color='blue')
    if train_acc_filtered:
        axes[0].plot(max_turns_filtered, train_acc_filtered, marker='s', linewidth=2,
                     markersize=8, label='Train Accuracy', color='green', linestyle='--')

    axes[0].set_xlabel("Max Turns (Limit)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Accuracy", fontsize=12, fontweight='bold')
    axes[0].set_title("Max Turns vs Accuracy", fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])
    if train_acc_filtered:
        axes[0].legend()

    # Add value annotations
    for x, y in zip(max_turns_filtered, test_acc_filtered):
        axes[0].annotate(f'{y:.2f}', xy=(x, y), xytext=(0, 5),
                        textcoords='offset points', ha='center', fontsize=9)

    # Plot 2: Average turns taken vs test accuracy
    axes[1].plot(avg_turns_filtered, test_acc_filtered, marker='o', linewidth=2,
                 markersize=8, label='Test Accuracy', color='blue')
    if train_acc_filtered:
        axes[1].plot(avg_turns_filtered, train_acc_filtered, marker='s', linewidth=2,
                     markersize=8, label='Train Accuracy', color='green', linestyle='--')

    axes[1].set_xlabel("Average Turns Taken", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Accuracy", fontsize=12, fontweight='bold')
    axes[1].set_title("Average Turns Taken vs Accuracy", fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])
    if train_acc_filtered:
        axes[1].legend()

    # Add value annotations
    for x, y in zip(avg_turns_filtered, test_acc_filtered):
        axes[1].annotate(f'{y:.2f}', xy=(x, y), xytext=(0, 5),
                        textcoords='offset points', ha='center', fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def print_summary(
    max_turns_values: List[int],
    test_accuracies: List[float],
    avg_turns_taken: List[float],
    train_accuracies: List[float],
    mode: str
):
    """Print summary table of results."""
    print("\n" + "="*80)
    print(f"SWEEP RESULTS SUMMARY ({mode.upper()})")
    print("="*80)
    print(f"{'Max Turns':<12} {'Avg Turns':<12} {'Test Acc':<12} {'Train Acc':<12}")
    print("-"*80)

    for i, max_turns in enumerate(max_turns_values):
        test_acc = test_accuracies[i] if i < len(test_accuracies) else None
        avg_turns = avg_turns_taken[i] if i < len(avg_turns_taken) else None
        train_acc = train_accuracies[i] if i < len(train_accuracies) and train_accuracies else None

        test_acc_str = f"{test_acc:.3f}" if test_acc is not None else "N/A"
        avg_turns_str = f"{avg_turns:.2f}" if avg_turns is not None else "N/A"
        train_acc_str = f"{train_acc:.3f}" if train_acc is not None else "N/A"

        print(f"{max_turns:<12} {avg_turns_str:<12} {test_acc_str:<12} {train_acc_str:<12}")

    print("="*80)

    # Find best settings
    if test_accuracies and any(acc is not None for acc in test_accuracies):
        valid_test_accs = [(i, acc) for i, acc in enumerate(test_accuracies) if acc is not None]
        best_idx, best_acc = max(valid_test_accs, key=lambda x: x[1])
        print(f"\nBest test accuracy: {best_acc:.3f} at max_turns={max_turns_values[best_idx]}")
        print(f"Average turns taken: {avg_turns_taken[best_idx]:.2f}")

    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser("Plot max_turns sweep results")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing sweep_results.json")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default=None,
                        help="Which results to plot (defaults to auto-detect)")
    parser.add_argument("--save_plot", type=str, default=None,
                        help="Path to save the plot (if not specified, shows interactive plot)")
    parser.add_argument("--no_summary", action="store_true",
                        help="Skip printing summary table")

    args = parser.parse_args()

    # Load results
    print(f"Loading sweep results from {args.results_dir}...")
    sweep_results = load_sweep_results(args.results_dir)

    # Auto-detect mode if not specified
    if args.mode is None:
        if "eval" in sweep_results:
            args.mode = "eval"
        elif "train" in sweep_results:
            args.mode = "train"
        else:
            raise ValueError("Could not auto-detect mode. Please specify --mode")

    print(f"Using mode: {args.mode}")

    # Extract metrics
    max_turns_values, test_accuracies, avg_turns_taken, train_accuracies = extract_metrics(
        sweep_results, args.mode
    )

    if not max_turns_values:
        print("Error: No valid results found!")
        return

    # Print summary
    if not args.no_summary:
        print_summary(max_turns_values, test_accuracies, avg_turns_taken, train_accuracies, args.mode)

    # Determine output path
    output_path = args.save_plot
    if output_path is None and args.results_dir:
        output_path = os.path.join(args.results_dir, f"sweep_{args.mode}_results.png")

    # Plot results
    plot_sweep_results(
        max_turns_values,
        test_accuracies,
        avg_turns_taken,
        train_accuracies,
        args.mode,
        output_path
    )


if __name__ == "__main__":
    main()
