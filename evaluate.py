#!/usr/bin/env python3
"""
Evaluate a model on AgentClinic scenarios without fine-tuning.

This script provides a baseline for comparison with PPO fine-tuned models.

Usage:
    # Evaluate base model on all scenarios
    python evaluate.py --base_model_name HF_Qwen/Qwen2.5-7B-Instruct

    # Evaluate on specific subset
    python evaluate.py --base_model_name HF_Qwen/Qwen2.5-7B-Instruct --max_scenarios 10

    # Evaluate with train/test split
    python evaluate.py --base_model_name HF_Qwen/Qwen2.5-7B-Instruct --test_size 20

    # Evaluate a fine-tuned checkpoint
    python evaluate.py --model_name path/to/checkpoint --test_size 20
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, set_seed, AutoModelForCausalLM

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None

# Import from ppo.py
from ppo import (
    ScenarioLoaderMedQA,
    AgentClinicSimulator,
    create_bnb_config,
)

logger = logging.getLogger(__name__)


def load_model_for_evaluation(
    model_name: str,
    bnb_config: Optional[BitsAndBytesConfig],
    torch_dtype: Optional[torch.dtype],
    device: str,
    trust_remote_code: bool,
):
    """Load model for evaluation (no value head needed)."""
    model_name = model_name.replace("HF_", "")
    device_map = "auto" if device == "auto" else {"": device}

    logger.info("Loading model from %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model


def evaluate(args) -> None:
    """Run evaluation without training."""
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    set_seed(args.seed)

    # Load scenarios
    scenario_loader = ScenarioLoaderMedQA(
        path=args.dataset_path,
        max_scenarios=args.max_scenarios,
        test_size=args.test_size,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    if scenario_loader.num_scenarios == 0:
        raise ValueError("No scenarios loaded; verify --dataset_path and format.")

    train_indices = scenario_loader.get_train_indices()
    test_indices = scenario_loader.get_test_indices()

    # Create simulator
    simulator = AgentClinicSimulator(
        scenario_loader=scenario_loader,
        patient_backend=args.patient_llm,
        measurement_backend=args.measurement_llm,
        moderator_backend=args.moderator_llm,
        max_turns=args.max_turns,
        question_cost=args.question_cost,
        temporal_decay_beta=args.temporal_decay_beta,
        budget_reward_weight=args.budget_reward_weight,
        question_reward_weight=args.question_reward_weight,
        diagnosis_reward_weight=args.diagnosis_reward_weight,
        seed=args.seed,
        debug_print=args.debug_print,
        reward_breakdown_debug=args.print_reward_breakdown,
        reward_forward_sim=args.reward_forward_sim,
        intrinsic_token_weight=args.intrinsic_token_weight,
        intrinsic_turn_weight=args.intrinsic_turn_weight,
        forward_sim_temperature=args.forward_sim_temperature,
    )

    # Load model
    bnb_config = create_bnb_config(args)
    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)

    model_source = args.model_name or args.base_model_name
    model = load_model_for_evaluation(
        model_source,
        bnb_config,
        torch_dtype,
        args.device,
        args.trust_remote_code,
    )

    tokenizer_name = args.tokenizer_name or args.base_model_name
    tokenizer_name = tokenizer_name.replace("HF_", "")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=not args.disable_fast_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )

    if tokenizer.pad_token is None:
        pad_token = tokenizer.eos_token or "<|pad|>"
        tokenizer.add_special_tokens({"pad_token": pad_token})
        model.resize_token_embeddings(len(tokenizer))

    tokenizer.padding_side = "left"

    device = torch.device(args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")

    # Generation kwargs
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": args.temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    def generate_response(
        prompt: Tuple[str, str],
        scenario_id: Optional[int] = None,
        turn_idx: Optional[int] = None,
    ) -> Tuple[torch.LongTensor, torch.LongTensor, str]:
        """Generate response from model."""
        system_prompt_text, user_prompt_text = prompt

        chat_messages: List[Dict[str, str]] = []
        if system_prompt_text:
            chat_messages.append({"role": "system", "content": system_prompt_text})
        chat_messages.append({"role": "user", "content": user_prompt_text})

        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            prompt_for_model = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_for_model = (
                f"{system_prompt_text}\n\n{user_prompt_text}"
                if system_prompt_text
                else user_prompt_text
            )

        inputs = tokenizer(prompt_for_model, return_tensors="pt").to(device)
        query_tensors = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        with torch.no_grad():
            output_tensors = model.generate(
                query_tensors,
                attention_mask=attention_mask,
                **generation_kwargs,
            )

        generated_tokens = output_tensors[:, query_tensors.shape[-1]:]
        if generated_tokens.shape[-1] == 0:
            generated_tokens = output_tensors[:, -1:]

        query_tensor = query_tensors.squeeze(0).detach()
        response_tensor = generated_tokens.squeeze(0).detach()
        response_text = tokenizer.decode(response_tensor, skip_special_tokens=True).strip()

        return query_tensor, response_tensor, response_text

    results = {}

    # Evaluate on training set
    if train_indices and args.eval_train:
        logger.info("Evaluating on %d training scenarios...", len(train_indices))
        train_metrics: List[Dict[str, float]] = []
        train_rewards: List[float] = []

        for scenario_idx in train_indices:
            state, episode_info = simulator.run_episode(scenario_idx, generate_response)
            train_metrics.append(
                {
                    "correctness": episode_info.get("correctness", 0.0),
                    "num_turns": len(state.turns),
                }
            )
            train_rewards.append(episode_info.get("reward", 0.0))

        total_correct = sum(m["correctness"] for m in train_metrics)
        train_accuracy = total_correct / len(train_metrics)
        train_avg_turns = sum(m["num_turns"] for m in train_metrics) / len(train_metrics)
        train_avg_reward = sum(train_rewards) / len(train_rewards)

        logger.info(
            "[TRAIN] accuracy=%.3f (%d/%d), avg_interactions=%.2f, avg_reward=%.3f",
            train_accuracy,
            int(total_correct),
            len(train_metrics),
            train_avg_turns,
            train_avg_reward,
        )

        results["train"] = {
            "accuracy": train_accuracy,
            "avg_interactions": train_avg_turns,
            "avg_reward": train_avg_reward,
            "num_scenarios": len(train_indices),
            "num_correct": int(total_correct),
        }

    # Evaluate on test set
    if test_indices:
        logger.info("Evaluating on %d test scenarios...", len(test_indices))
        test_metrics: List[Dict[str, float]] = []
        test_rewards: List[float] = []

        for scenario_idx in test_indices:
            state, episode_info = simulator.run_episode(scenario_idx, generate_response)
            test_metrics.append(
                {
                    "correctness": episode_info.get("correctness", 0.0),
                    "num_turns": len(state.turns),
                }
            )
            test_rewards.append(episode_info.get("reward", 0.0))

        total_correct = sum(m["correctness"] for m in test_metrics)
        test_accuracy = total_correct / len(test_metrics)
        test_avg_turns = sum(m["num_turns"] for m in test_metrics) / len(test_metrics)
        test_avg_reward = sum(test_rewards) / len(test_rewards)

        logger.info(
            "[TEST] accuracy=%.3f (%d/%d), avg_interactions=%.2f, avg_reward=%.3f",
            test_accuracy,
            int(total_correct),
            len(test_metrics),
            test_avg_turns,
            test_avg_reward,
        )

        results["test"] = {
            "accuracy": test_accuracy,
            "avg_interactions": test_avg_turns,
            "avg_reward": test_avg_reward,
            "num_scenarios": len(test_indices),
            "num_correct": int(total_correct),
        }

    # Evaluate on all scenarios if no split
    if not test_indices and not args.eval_train:
        logger.info("Evaluating on all %d scenarios...", scenario_loader.num_scenarios)
        all_metrics: List[Dict[str, float]] = []
        all_rewards: List[float] = []

        for scenario_idx in range(scenario_loader.num_scenarios):
            state, episode_info = simulator.run_episode(scenario_idx, generate_response)
            all_metrics.append(
                {
                    "correctness": episode_info.get("correctness", 0.0),
                    "num_turns": len(state.turns),
                }
            )
            all_rewards.append(episode_info.get("reward", 0.0))

        total_correct = sum(m["correctness"] for m in all_metrics)
        accuracy = total_correct / len(all_metrics)
        avg_turns = sum(m["num_turns"] for m in all_metrics) / len(all_metrics)
        avg_reward = sum(all_rewards) / len(all_rewards)

        logger.info(
            "[ALL] accuracy=%.3f (%d/%d), avg_interactions=%.2f, avg_reward=%.3f",
            accuracy,
            int(total_correct),
            len(all_metrics),
            avg_turns,
            avg_reward,
        )

        results["all"] = {
            "accuracy": accuracy,
            "avg_interactions": avg_turns,
            "avg_reward": avg_reward,
            "num_scenarios": scenario_loader.num_scenarios,
            "num_correct": int(total_correct),
        }

    # Save results
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved evaluation results to %s", results_path)

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    for split, metrics in results.items():
        print(f"\n{split.upper()}:")
        print(f"  Accuracy: {metrics['accuracy']:.3f} ({metrics['num_correct']}/{metrics['num_scenarios']})")
        print(f"  Avg interactions: {metrics['avg_interactions']:.2f}")
        print(f"  Avg reward: {metrics['avg_reward']:.3f}")
    print("=" * 70 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("AgentClinic baseline evaluation")
    parser.add_argument("--dataset_path", type=str, default="agentclinic_medqa.jsonl")
    parser.add_argument("--max_scenarios", type=int, default=None)
    parser.add_argument("--test_size", type=int, default=None)
    parser.add_argument("--test_ratio", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/baseline_eval")
    parser.add_argument("--base_model_name", type=str, default="HF_Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model_name", type=str, default=None, help="Evaluate a specific checkpoint instead of base model")
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--eval_train", action="store_true", help="Also evaluate on training set")

    parser.add_argument("--patient_llm", type=str, default="HF_Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--measurement_llm", type=str, default="HF_Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--moderator_llm", type=str, default="HF_Qwen/Qwen2.5-7B-Instruct")

    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--disable_fast_tokenizer", action="store_true")

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--question_cost", type=float, default=1.0)
    parser.add_argument("--temporal_decay_beta", type=float, default=1.0)
    parser.add_argument("--budget_reward_weight", type=float, default=1.0)
    parser.add_argument("--question_reward_weight", type=float, default=1.0)
    parser.add_argument("--diagnosis_reward_weight", type=float, default=1.0)
    parser.add_argument("--debug_print", action="store_true")
    parser.add_argument("--print_reward_breakdown", action="store_true")
    parser.add_argument("--reward_forward_sim", action="store_true", help="Use forward simulation rewards: per-turn extrinsic (forward sim correctness) + intrinsic (token/turn penalties)")
    parser.add_argument("--intrinsic_token_weight", type=float, default=0.001, help="Penalty weight per token in intrinsic reward")
    parser.add_argument("--intrinsic_turn_weight", type=float, default=0.0, help="Flat penalty per turn in intrinsic reward")
    parser.add_argument("--forward_sim_temperature", type=float, default=0.0, help="Temperature for forward simulation (0.0 = deterministic)")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", type=str, default="INFO")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.bf16 and args.fp16:
        raise ValueError("Only one of --bf16 or --fp16 can be specified.")
    if args.test_size is not None and args.test_ratio is not None:
        raise ValueError("Cannot specify both --test_size and --test_ratio; choose one.")

    evaluate(args)


if __name__ == "__main__":
    main()
