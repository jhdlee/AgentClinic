#!/usr/bin/env python3
"""
Sweep over max_turns parameter and evaluate performance.

This script runs PPO training and/or evaluation for different values of max_turns,
then aggregates the results for comparison.

Usage:
    # Training sweep
    python sweep_max_turns.py --mode train --max_turns_range 2 3 4 5

    # Evaluation sweep (baseline)
    python sweep_max_turns.py --mode eval --max_turns_range 2 3 4 5

    # Both training and evaluation
    python sweep_max_turns.py --mode both --max_turns_range 2 3 4 5
"""

import argparse
import json
import os
import subprocess
import sys
from typing import List, Dict, Any


def run_command(cmd: List[str], description: str) -> int:
    """Run a command and return the exit code."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    return result.returncode


def get_run_name(args, max_turns: int, mode: str) -> str:
    """Generate informative run name based on hyperparameters."""
    parts = [
        mode,
        f"mt{max_turns}",  # max_turns
    ]

    if mode == "train":
        parts.extend([
            f"lr{args.learning_rate}",
            f"ep{args.num_train_epochs}",
        ])

        if args.reward_forward_sim:
            parts.append("fwd_sim")
            if args.intrinsic_token_weight != 0.001:  # Only add if non-default
                parts.append(f"tw{args.intrinsic_token_weight}")
            if args.intrinsic_turn_weight != 0.0:  # Only add if non-default
                parts.append(f"turw{args.intrinsic_turn_weight}")

        if args.use_lora:
            parts.append("lora")
        if args.use_4bit:
            parts.append("4bit")

    return "_".join(parts)


def sweep_training(args, max_turns_values: List[int]) -> Dict[int, Dict[str, Any]]:
    """Run training sweep over max_turns values."""
    results = {}

    for max_turns in max_turns_values:
        run_name = get_run_name(args, max_turns, "train")
        output_dir = os.path.join(args.base_output_dir, run_name)

        cmd = [
            "python3", "ppo.py",
            "--dataset_path", args.dataset_path,
            "--output_dir", output_dir,
            "--base_model_name", args.base_model_name,
            "--patient_llm", args.patient_llm,
            "--measurement_llm", args.measurement_llm,
            "--moderator_llm", args.moderator_llm,
            "--max_turns", str(max_turns),
            "--num_train_epochs", str(args.num_train_epochs),
            "--learning_rate", str(args.learning_rate),
            "--diagnosis_reward_weight", str(args.diagnosis_reward_weight),
            "--budget_reward_weight", str(args.budget_reward_weight),
            "--question_reward_weight", str(args.question_reward_weight),
            "--seed", str(args.seed),
        ]

        if args.max_scenarios:
            cmd.extend(["--max_scenarios", str(args.max_scenarios)])
        if args.test_size:
            cmd.extend(["--test_size", str(args.test_size)])
        if args.test_ratio:
            cmd.extend(["--test_ratio", str(args.test_ratio)])
        if args.use_lora:
            cmd.append("--use_lora")
        if args.use_4bit:
            cmd.append("--use_4bit")
        if args.disable_reference_model:
            cmd.append("--disable_reference_model")
        if args.reward_forward_sim:
            cmd.append("--reward_forward_sim")
            cmd.extend(["--intrinsic_token_weight", str(args.intrinsic_token_weight)])
            cmd.extend(["--intrinsic_turn_weight", str(args.intrinsic_turn_weight)])
            cmd.extend(["--forward_sim_temperature", str(args.forward_sim_temperature)])

        exit_code = run_command(
            cmd,
            f"Training with max_turns={max_turns} ({run_name})"
        )

        if exit_code != 0:
            print(f"ERROR: Training failed for max_turns={max_turns}")
            results[max_turns] = {"status": "failed", "output_dir": output_dir}
            if not args.continue_on_error:
                break
        else:
            # Load epoch stats
            epoch_stats_path = os.path.join(output_dir, "epoch_stats.json")
            if os.path.exists(epoch_stats_path):
                with open(epoch_stats_path, "r") as f:
                    epoch_stats = json.load(f)
                    final_epoch = epoch_stats[-1] if epoch_stats else {}
            else:
                final_epoch = {}

            results[max_turns] = {
                "status": "success",
                "output_dir": output_dir,
                "run_name": run_name,
                "final_epoch": final_epoch,
            }

    return results


def sweep_evaluation(args, max_turns_values: List[int]) -> Dict[int, Dict[str, Any]]:
    """Run evaluation sweep over max_turns values."""
    results = {}

    for max_turns in max_turns_values:
        run_name = get_run_name(args, max_turns, "eval")
        output_dir = os.path.join(args.base_output_dir, run_name)

        cmd = [
            "python3", "evaluate.py",
            "--dataset_path", args.dataset_path,
            "--output_dir", output_dir,
            "--base_model_name", args.base_model_name,
            "--patient_llm", args.patient_llm,
            "--measurement_llm", args.measurement_llm,
            "--moderator_llm", args.moderator_llm,
            "--max_turns", str(max_turns),
            "--seed", str(args.seed),
        ]

        if args.max_scenarios:
            cmd.extend(["--max_scenarios", str(args.max_scenarios)])
        if args.test_size:
            cmd.extend(["--test_size", str(args.test_size)])
        if args.test_ratio:
            cmd.extend(["--test_ratio", str(args.test_ratio)])
        if args.use_4bit:
            cmd.append("--use_4bit")
        if args.eval_train:
            cmd.append("--eval_train")
        if args.reward_forward_sim:
            cmd.append("--reward_forward_sim")
            cmd.extend(["--intrinsic_token_weight", str(args.intrinsic_token_weight)])
            cmd.extend(["--intrinsic_turn_weight", str(args.intrinsic_turn_weight)])
            cmd.extend(["--forward_sim_temperature", str(args.forward_sim_temperature)])

        exit_code = run_command(
            cmd,
            f"Evaluating with max_turns={max_turns} ({run_name})"
        )

        if exit_code != 0:
            print(f"ERROR: Evaluation failed for max_turns={max_turns}")
            results[max_turns] = {"status": "failed", "output_dir": output_dir}
            if not args.continue_on_error:
                break
        else:
            # Load evaluation results
            eval_results_path = os.path.join(output_dir, "evaluation_results.json")
            if os.path.exists(eval_results_path):
                with open(eval_results_path, "r") as f:
                    eval_results = json.load(f)
            else:
                eval_results = {}

            results[max_turns] = {
                "status": "success",
                "output_dir": output_dir,
                "run_name": run_name,
                "eval_results": eval_results,
            }

    return results


def save_sweep_results(results: Dict[int, Dict[str, Any]], output_path: str):
    """Save sweep results to JSON file."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved sweep results to {output_path}")


def main():
    parser = argparse.ArgumentParser("Sweep over max_turns parameter")

    # Sweep parameters
    parser.add_argument("--mode", type=str, choices=["train", "eval", "both"], required=True,
                        help="Whether to run training, evaluation, or both")
    parser.add_argument("--max_turns_range", type=int, nargs="+", required=True,
                        help="List of max_turns values to sweep over (e.g., 2 3 4 5)")
    parser.add_argument("--base_output_dir", type=str, default="outputs/sweep_max_turns",
                        help="Base directory for all sweep outputs")
    parser.add_argument("--continue_on_error", action="store_true",
                        help="Continue sweep even if one run fails")

    # Dataset parameters
    parser.add_argument("--dataset_path", type=str, default="agentclinic_medqa.jsonl")
    parser.add_argument("--max_scenarios", type=int, default=None)
    parser.add_argument("--test_size", type=int, default=None)
    parser.add_argument("--test_ratio", type=float, default=None)

    # Model parameters
    parser.add_argument("--base_model_name", type=str, default="HF_Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--patient_llm", type=str, default="HF_Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--measurement_llm", type=str, default="HF_Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--moderator_llm", type=str, default="HF_Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--disable_reference_model", action="store_true")

    # Training parameters
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--diagnosis_reward_weight", type=float, default=1.0)
    parser.add_argument("--budget_reward_weight", type=float, default=1.0)
    parser.add_argument("--question_reward_weight", type=float, default=1.0)
    parser.add_argument("--reward_forward_sim", action="store_true", help="Use forward simulation rewards")
    parser.add_argument("--intrinsic_token_weight", type=float, default=0.001, help="Penalty weight per token")
    parser.add_argument("--intrinsic_turn_weight", type=float, default=0.0, help="Flat penalty per turn")
    parser.add_argument("--forward_sim_temperature", type=float, default=0.0, help="Temperature for forward simulation")

    # Evaluation parameters
    parser.add_argument("--eval_train", action="store_true")

    # Other
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    os.makedirs(args.base_output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"MAX_TURNS SWEEP")
    print(f"{'='*70}")
    print(f"Mode: {args.mode}")
    print(f"Max turns values: {args.max_turns_range}")
    print(f"Output directory: {args.base_output_dir}")
    print(f"{'='*70}\n")

    all_results = {}

    # Run training sweep
    if args.mode in ["train", "both"]:
        print("\n" + "="*70)
        print("TRAINING SWEEP")
        print("="*70)
        train_results = sweep_training(args, args.max_turns_range)
        all_results["train"] = train_results

        # Save intermediate results
        save_sweep_results(
            train_results,
            os.path.join(args.base_output_dir, "train_sweep_results.json")
        )

    # Run evaluation sweep
    if args.mode in ["eval", "both"]:
        print("\n" + "="*70)
        print("EVALUATION SWEEP")
        print("="*70)
        eval_results = sweep_evaluation(args, args.max_turns_range)
        all_results["eval"] = eval_results

        # Save intermediate results
        save_sweep_results(
            eval_results,
            os.path.join(args.base_output_dir, "eval_sweep_results.json")
        )

    # Save combined results
    save_sweep_results(
        all_results,
        os.path.join(args.base_output_dir, "sweep_results.json")
    )

    print("\n" + "="*70)
    print("SWEEP COMPLETE")
    print("="*70)
    print(f"Results saved to: {args.base_output_dir}")
    print(f"\nTo plot results, run:")
    print(f"  python plot_sweep.py --results_dir {args.base_output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
