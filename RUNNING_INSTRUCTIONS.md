# Running Instructions for AgentClinic PPO Training

This guide covers how to run baseline evaluation, forward simulation reward training, and hyperparameter sweeps.

---

## Table of Contents
1. [Baseline Evaluation (No Fine-tuning)](#1-baseline-evaluation-no-fine-tuning)
2. [Training with Forward Simulation Rewards](#2-training-with-forward-simulation-rewards)
3. [Hyperparameter Sweeps](#3-hyperparameter-sweeps)
4. [Key Parameters Explained](#4-key-parameters-explained)

---

## 1. Baseline Evaluation (No Fine-tuning)

The baseline evaluation runs the model on AgentClinic scenarios **without any PPO training**. This provides a reference point for comparison.

### Basic Usage

```bash
python3 evaluate.py \
  --base_model_name HF_Qwen/Qwen2.5-7B-Instruct \
  --dataset_path agentclinic_medqa.jsonl \
  --output_dir outputs/baseline_eval
```

### With Train/Test Split

```bash
python3 evaluate.py \
  --base_model_name HF_Qwen/Qwen2.5-7B-Instruct \
  --dataset_path agentclinic_medqa.jsonl \
  --test_size 20 \
  --output_dir outputs/baseline_eval
```

### Important Baseline Parameters

```bash
python3 evaluate.py \
  --base_model_name HF_Qwen/Qwen2.5-7B-Instruct \
  --patient_llm HF_Qwen/Qwen2.5-7B-Instruct \
  --measurement_llm HF_Qwen/Qwen2.5-7B-Instruct \
  --moderator_llm HF_Qwen/Qwen2.5-7B-Instruct \
  --max_turns 5 \
  --max_scenarios 50 \
  --test_size 10 \
  --temperature 0.0 \
  --seed 42
```

### Baseline Output

Results are saved to `{output_dir}/evaluation_results.json`:
```json
{
  "test": {
    "accuracy": 0.650,
    "avg_interactions": 4.5,
    "avg_reward": 0.0,
    "num_scenarios": 10,
    "num_correct": 6
  }
}
```

---

## 2. Training with Forward Simulation Rewards

The `reward_forward_sim` method trains the model using per-turn credit assignment via forward simulation.

### Basic Training

```bash
python3 ppo.py \
  --base_model_name HF_Qwen/Qwen2.5-7B-Instruct \
  --dataset_path agentclinic_medqa.jsonl \
  --output_dir outputs/ppo_forward_sim \
  --reward_forward_sim \
  --num_train_epochs 3 \
  --max_turns 5 \
  --learning_rate 1e-6
```

### Full Training Example with All Options

```bash
python3 ppo.py \
  --base_model_name HF_Qwen/Qwen2.5-7B-Instruct \
  --patient_llm HF_Qwen/Qwen2.5-7B-Instruct \
  --measurement_llm HF_Qwen/Qwen2.5-7B-Instruct \
  --moderator_llm HF_Qwen/Qwen2.5-7B-Instruct \
  --dataset_path agentclinic_medqa.jsonl \
  --output_dir outputs/ppo_forward_sim \
  --max_scenarios 50 \
  --test_size 10 \
  --reward_forward_sim \
  --diagnosis_reward_weight 1.0 \
  --intrinsic_token_weight 0.001 \
  --intrinsic_turn_weight 0.0 \
  --forward_sim_temperature 0.0 \
  --num_train_epochs 3 \
  --learning_rate 1e-6 \
  --max_turns 5 \
  --use_lora \
  --peft_r 32 \
  --peft_alpha 16 \
  --seed 42
```

### Training with LoRA and 4-bit Quantization (Memory Efficient)

```bash
python3 ppo.py \
  --base_model_name HF_Qwen/Qwen2.5-7B-Instruct \
  --dataset_path agentclinic_medqa.jsonl \
  --output_dir outputs/ppo_lora_4bit \
  --reward_forward_sim \
  --use_lora \
  --use_4bit \
  --num_train_epochs 3 \
  --max_turns 5
```

### Training Output

After training completes, you'll find:
- `{output_dir}/evaluation_results.json` - Final accuracy/metrics on train/test sets
- `{output_dir}/epoch_stats.json` - Per-epoch training statistics
- Model checkpoints and weights

---

## 3. Hyperparameter Sweeps

### Sweep Script Overview

The [sweep_max_turns.py](sweep_max_turns.py) script automates running experiments across different hyperparameter values.

### A. Baseline Sweep (Evaluation Only)

Sweep different `max_turns` values **without training**:

```bash
python3 sweep_max_turns.py \
  --mode eval \
  --max_turns_range 2 3 4 5 \
  --base_output_dir outputs/sweep_baseline \
  --base_model_name HF_Qwen/Qwen2.5-7B-Instruct \
  --max_scenarios 50 \
  --test_size 10
```

### B. Forward Simulation Training Sweep

Sweep different `max_turns` values **with forward simulation training**:

```bash
python3 sweep_max_turns.py \
  --mode train \
  --max_turns_range 2 3 4 5 \
  --base_output_dir outputs/sweep_fwd_sim \
  --base_model_name HF_Qwen/Qwen2.5-7B-Instruct \
  --reward_forward_sim \
  --intrinsic_token_weight 0.001 \
  --intrinsic_turn_weight 0.0 \
  --num_train_epochs 3 \
  --learning_rate 1e-5 \
  --max_scenarios 50 \
  --test_size 10
```

### C. Combined Sweep (Train + Eval)

Run both training and evaluation in one sweep:

```bash
python3 sweep_max_turns.py \
  --mode both \
  --max_turns_range 2 3 4 5 \
  --base_output_dir outputs/sweep_both \
  --reward_forward_sim \
  --num_train_epochs 3 \
  --use_lora \
  --max_scenarios 50 \
  --test_size 10
```

### D. Sweep with LoRA and 4-bit (Memory Efficient)

```bash
python3 sweep_max_turns.py \
  --mode train \
  --max_turns_range 2 3 4 5 \
  --base_output_dir outputs/sweep_efficient \
  --reward_forward_sim \
  --use_lora \
  --use_4bit \
  --num_train_epochs 3 \
  --learning_rate 1e-5
```

### Sweep Output Structure

After running a sweep, results are organized as:

```
outputs/sweep_baseline/
├── eval_mt2/                           # max_turns=2 evaluation
│   └── evaluation_results.json
├── eval_mt3/                           # max_turns=3 evaluation
│   └── evaluation_results.json
├── eval_mt4/                           # max_turns=4 evaluation
│   └── evaluation_results.json
├── eval_mt5/                           # max_turns=5 evaluation
│   └── evaluation_results.json
├── eval_sweep_results.json             # Aggregated eval results
└── sweep_results.json                  # Complete sweep summary
```

For training sweeps:

```
outputs/sweep_fwd_sim/
├── train_mt2_lr1e-05_ep3_fwd_sim/      # max_turns=2 training
│   ├── evaluation_results.json
│   └── epoch_stats.json
├── train_mt3_lr1e-05_ep3_fwd_sim/      # max_turns=3 training
│   ├── evaluation_results.json
│   └── epoch_stats.json
├── train_sweep_results.json             # Aggregated training results
└── sweep_results.json                   # Complete sweep summary
```

---

## 4. Key Parameters Explained

### Model Parameters

- `--base_model_name`: HuggingFace model to use (prefix with `HF_`)
- `--patient_llm`: Model for patient agent (default: same as base)
- `--measurement_llm`: Model for measurement/test agent
- `--moderator_llm`: Model for diagnosis evaluation

### Dataset Parameters

- `--dataset_path`: Path to JSONL dataset (default: `agentclinic_medqa.jsonl`)
- `--max_scenarios`: Limit number of scenarios (useful for quick tests)
- `--test_size`: Number of scenarios for test set (e.g., 10)
- `--test_ratio`: Ratio of scenarios for test set (e.g., 0.2 for 20%)

### Simulation Parameters

- `--max_turns`: Maximum doctor-patient interactions before forced diagnosis
- `--question_cost`: Budget cost per question (default: 1.0)
- `--temperature`: Sampling temperature (0.0 = deterministic)

### Forward Simulation Reward Parameters

- `--reward_forward_sim`: Enable forward simulation rewards (**required for training**)
- `--diagnosis_reward_weight`: Weight for diagnosis correctness (default: 1.0)
- `--intrinsic_token_weight`: Penalty per token generated (default: 0.001)
- `--intrinsic_turn_weight`: Penalty per turn taken (default: 0.0)
- `--forward_sim_temperature`: Temperature for forward simulation (default: 0.0)

### Training Parameters

- `--num_train_epochs`: Number of epochs to train (default: 3)
- `--learning_rate`: Learning rate (default: 1e-6)
- `--num_ppo_epochs`: PPO optimization epochs per batch (default: 4)

### Memory Efficiency Parameters

- `--use_lora`: Enable LoRA adapters (recommended)
- `--use_4bit`: Use 4-bit quantization (saves memory)
- `--gradient_checkpointing`: Enable gradient checkpointing
- `--disable_reference_model`: Skip reference model (saves memory, disables KL)

### Sweep Parameters

- `--mode`: Sweep mode (`train`, `eval`, or `both`)
- `--max_turns_range`: List of max_turns values to sweep (e.g., `2 3 4 5`)
- `--base_output_dir`: Base directory for sweep outputs
- `--continue_on_error`: Continue sweep if one run fails

---

## Quick Start Examples

### Example 1: Quick Baseline Test

```bash
# Evaluate baseline on 10 scenarios
python3 evaluate.py \
  --max_scenarios 10 \
  --output_dir outputs/quick_baseline
```

### Example 2: Quick Training Test

```bash
# Train with forward sim on 10 scenarios for 1 epoch
python3 ppo.py \
  --max_scenarios 10 \
  --test_size 2 \
  --reward_forward_sim \
  --num_train_epochs 1 \
  --output_dir outputs/quick_train
```

### Example 3: Full Experiment Comparison

```bash
# Step 1: Baseline evaluation
python3 evaluate.py \
  --max_scenarios 50 \
  --test_size 10 \
  --output_dir outputs/exp1_baseline

# Step 2: Train with forward simulation
python3 ppo.py \
  --max_scenarios 50 \
  --test_size 10 \
  --reward_forward_sim \
  --num_train_epochs 3 \
  --use_lora \
  --output_dir outputs/exp1_trained

# Step 3: Compare results
cat outputs/exp1_baseline/evaluation_results.json
cat outputs/exp1_trained/evaluation_results.json
```

### Example 4: Comprehensive Sweep

```bash
# Sweep both baseline and training across max_turns
python3 sweep_max_turns.py \
  --mode both \
  --max_turns_range 3 4 5 \
  --max_scenarios 50 \
  --test_size 10 \
  --reward_forward_sim \
  --num_train_epochs 3 \
  --use_lora \
  --base_output_dir outputs/comprehensive_sweep

# Plot results
python3 plot_sweep.py --results_dir outputs/comprehensive_sweep
```

---

## Notes

1. **Forward simulation is required for training**: You must use `--reward_forward_sim` when running `ppo.py`. The older reward modes have been archived.

2. **Baseline vs Training**:
   - Use `evaluate.py` for baseline (no training)
   - Use `ppo.py` for training with forward simulation

3. **Memory considerations**:
   - For large models, use `--use_lora --use_4bit`
   - Consider `--disable_reference_model` if memory is tight

4. **Reproducibility**: Always set `--seed 42` for reproducible results

5. **Sweep organization**: Each sweep run creates a descriptive directory name based on hyperparameters (e.g., `train_mt5_lr1e-05_ep3_fwd_sim_lora`)
