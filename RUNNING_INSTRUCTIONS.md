# Running Instructions for AgentClinic PPO Training

This guide covers how to run baseline evaluation, forward simulation reward training, budget-aware reward training, and hyperparameter sweeps.

---

## Table of Contents
1. [vLLM Acceleration (Optional)](#1-vllm-acceleration-optional)
2. [Baseline Evaluation (No Fine-tuning)](#2-baseline-evaluation-no-fine-tuning)
3. [Training with Forward Simulation Rewards](#3-training-with-forward-simulation-rewards)
4. [Training with Budget-Aware Rewards](#4-training-with-budget-aware-rewards)
5. [Hyperparameter Sweeps](#5-hyperparameter-sweeps)
6. [Plotting Sweep Results](#6-plotting-sweep-results)
7. [Key Parameters Explained](#7-key-parameters-explained)

---

## 1. vLLM Acceleration (Optional)

vLLM provides **significantly faster inference** compared to standard HuggingFace models. It's especially beneficial for the patient, measurement, and moderator agents which perform many inference calls during training and evaluation.

### Installation

```bash
pip install vllm
```

### Usage

There are two ways to use vLLM:

#### A. For Agent Models (Patient, Measurement, Moderator)

Simply prefix the model name with `VLLM_` instead of `HF_`:

```bash
# Standard HuggingFace (slower)
--patient_llm HF_Qwen/Qwen2.5-7B-Instruct

# With vLLM acceleration (faster)
--patient_llm VLLM_Qwen/Qwen2.5-7B-Instruct
```

**Example with vLLM agents:**
```bash
python3 evaluate.py \
  --base_model_name HF_Qwen/Qwen2.5-7B-Instruct \
  --patient_llm VLLM_Qwen/Qwen2.5-7B-Instruct \
  --measurement_llm VLLM_Qwen/Qwen2.5-7B-Instruct \
  --moderator_llm VLLM_Qwen/Qwen2.5-7B-Instruct \
  --test_size 20
```

#### B. For Policy Model (Training and Evaluation)

Use the `--use_vllm_policy` flag:

**During Evaluation:**
```bash
python3 evaluate.py \
  --base_model_name Qwen/Qwen2.5-7B-Instruct \
  --use_vllm_policy \
  --patient_llm VLLM_Qwen/Qwen2.5-7B-Instruct \
  --test_size 20
```

**During Training:**
```bash
python3 ppo.py \
  --base_model_name HF_Qwen/Qwen2.5-7B-Instruct \
  --use_vllm_policy \
  --patient_llm VLLM_Qwen/Qwen2.5-7B-Instruct \
  --measurement_llm VLLM_Qwen/Qwen2.5-7B-Instruct \
  --moderator_llm VLLM_Qwen/Qwen2.5-7B-Instruct \
  --reward_forward_sim \
  --num_train_epochs 3
```

**Note:** When using `--use_vllm_policy` during training, vLLM is used only for generation (forward passes). The HuggingFace model is still loaded for computing gradients and performing PPO updates.

### vLLM Configuration Parameters

- `--vllm_tensor_parallel_size`: Number of GPUs for tensor parallelism (default: 1)
- `--vllm_gpu_memory_utilization`: GPU memory utilization 0.0-1.0 (default: 0.9)
- `--vllm_verbose`: Enable verbose vLLM logging (default: suppressed for cleaner output)

**Example with multi-GPU:**
```bash
python3 evaluate.py \
  --base_model_name Qwen/Qwen2.5-7B-Instruct \
  --use_vllm_policy \
  --patient_llm VLLM_Qwen/Qwen2.5-7B-Instruct \
  --vllm_tensor_parallel_size 2 \
  --vllm_gpu_memory_utilization 0.85
```

### Mixed Usage

You can mix HuggingFace and vLLM models:

```bash
python3 ppo.py \
  --base_model_name HF_Qwen/Qwen2.5-7B-Instruct \
  --patient_llm VLLM_Qwen/Qwen2.5-7B-Instruct \
  --measurement_llm HF_Qwen/Qwen2.5-7B-Instruct \
  --moderator_llm VLLM_meta-llama/Llama-3-8B-Instruct \
  --reward_forward_sim \
  --num_train_epochs 3
```

### When to Use vLLM

✅ **Recommended for:**
- Patient, measurement, and moderator agents (always)
- Policy model during evaluation
- Policy model during training (for faster generation)
- Large-scale experiments with many scenarios

💡 **How it works in training:**
- vLLM handles generation (forward passes) for 2-5x speedup
- HuggingFace model still computes gradients and performs PPO updates
- Both models loaded in parallel (requires sufficient GPU memory)

### Performance Benefits

Using vLLM for agent models typically provides:
- **2-5x faster inference** compared to HuggingFace
- Better GPU utilization with tensor parallelism
- Automatic batching and optimization

---

## 2. Baseline Evaluation (No Fine-tuning)

The baseline evaluation runs the model on AgentClinic scenarios **without any PPO training**. This provides a reference point for comparison.

**Note**: For baseline evaluation, the `--reward_forward_sim` flag is **optional**:
- **Without** `--reward_forward_sim`: Fast evaluation using simple correctness-based rewards (recommended for baseline)
- **With** `--reward_forward_sim`: Slower evaluation with forward simulation (useful for comparing reward signals)

### Basic Usage (Fast - Recommended)

```bash
python3 evaluate.py \
  --base_model_name HF_Qwen/Qwen2.5-7B-Instruct \
  --dataset_path agentclinic_medqa.jsonl \
  --output_dir outputs/baseline_eval
```

### With vLLM Acceleration (Faster)

```bash
python3 evaluate.py \
  --base_model_name HF_Qwen/Qwen2.5-7B-Instruct \
  --patient_llm VLLM_Qwen/Qwen2.5-7B-Instruct \
  --measurement_llm VLLM_Qwen/Qwen2.5-7B-Instruct \
  --moderator_llm VLLM_Qwen/Qwen2.5-7B-Instruct \
  --dataset_path agentclinic_medqa.jsonl \
  --output_dir outputs/baseline_eval_vllm
```

### With Train/Test Split

```bash
python3 evaluate.py \
  --base_model_name HF_Qwen/Qwen2.5-7B-Instruct \
  --dataset_path agentclinic_medqa.jsonl \
  --test_size 20 \
  --output_dir outputs/baseline_eval
```

### Full Baseline Parameters

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

## 3. Training with Forward Simulation Rewards

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

### With vLLM for Agent Models (Faster Training)

```bash
python3 ppo.py \
  --base_model_name HF_Qwen/Qwen2.5-7B-Instruct \
  --patient_llm VLLM_Qwen/Qwen2.5-7B-Instruct \
  --measurement_llm VLLM_Qwen/Qwen2.5-7B-Instruct \
  --moderator_llm VLLM_Qwen/Qwen2.5-7B-Instruct \
  --dataset_path agentclinic_medqa.jsonl \
  --output_dir outputs/ppo_forward_sim_vllm \
  --reward_forward_sim \
  --num_train_epochs 3 \
  --max_turns 5 \
  --learning_rate 1e-6
```

### With vLLM for Everything (Maximum Speed)

Use vLLM for both agent models AND policy model:

```bash
python3 ppo.py \
  --base_model_name HF_Qwen/Qwen2.5-7B-Instruct \
  --use_vllm_policy \
  --patient_llm VLLM_Qwen/Qwen2.5-7B-Instruct \
  --measurement_llm VLLM_Qwen/Qwen2.5-7B-Instruct \
  --moderator_llm VLLM_Qwen/Qwen2.5-7B-Instruct \
  --dataset_path agentclinic_medqa.jsonl \
  --output_dir outputs/ppo_forward_sim_vllm_full \
  --reward_forward_sim \
  --num_train_epochs 3 \
  --max_turns 5 \
  --learning_rate 1e-6 \
  --vllm_tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization 0.85
```

**Note:** When using `--use_vllm_policy`, both HuggingFace and vLLM models are loaded. Adjust `--vllm_gpu_memory_utilization` (e.g., 0.85 or lower) if you encounter OOM errors.

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

## 4. Training with Budget-Aware Rewards

The `reward_budget_aware` method trains the model to optimize turn efficiency while maintaining diagnostic accuracy. This reward mode encourages the model to use a target number of turns by:
- Computing confidence-based question utility (how much each question improves diagnostic confidence)
- Applying temporal weighting to earlier questions
- Adding rewards for staying below target turns or penalties for exceeding them

### Basic Budget-Aware Training

```bash
python3 ppo.py \
  --base_model_name HF_Qwen/Qwen2.5-7B-Instruct \
  --dataset_path agentclinic_medqa.jsonl \
  --output_dir outputs/ppo_budget_aware \
  --reward_budget_aware \
  --target_num_turns 3 \
  --turn_penalty_weight 0.1 \
  --turn_reward_weight 0.05 \
  --num_train_epochs 3 \
  --max_turns 5 \
  --learning_rate 1e-6
```

### Full Budget-Aware Training Example

```bash
python3 ppo.py \
  --base_model_name HF_Qwen/Qwen2.5-7B-Instruct \
  --patient_llm HF_Qwen/Qwen2.5-7B-Instruct \
  --measurement_llm HF_Qwen/Qwen2.5-7B-Instruct \
  --moderator_llm HF_Qwen/Qwen2.5-7B-Instruct \
  --dataset_path agentclinic_medqa.jsonl \
  --output_dir outputs/ppo_budget_aware_full \
  --max_scenarios 50 \
  --test_size 10 \
  --reward_budget_aware \
  --target_num_turns 3 \
  --turn_penalty_weight 0.1 \
  --turn_reward_weight 0.05 \
  --diagnosis_reward_weight 1.0 \
  --temporal_decay_beta 1.0 \
  --num_train_epochs 3 \
  --learning_rate 1e-6 \
  --max_turns 5 \
  --use_lora \
  --peft_r 32 \
  --peft_alpha 16 \
  --seed 42
```

### Budget-Aware with LoRA and 4-bit Quantization

```bash
python3 ppo.py \
  --base_model_name HF_Qwen/Qwen2.5-7B-Instruct \
  --dataset_path agentclinic_medqa.jsonl \
  --output_dir outputs/ppo_budget_aware_4bit \
  --reward_budget_aware \
  --target_num_turns 3 \
  --turn_penalty_weight 0.1 \
  --turn_reward_weight 0.05 \
  --use_lora \
  --use_4bit \
  --num_train_epochs 3 \
  --max_turns 5
```

### How Budget-Aware Rewards Work

1. **Question Utility**: For each question turn, the reward is based on how much the question increases diagnostic confidence:
   - Confidence is measured before and after each question
   - Utility = confidence_after - confidence_before
   - Temporal weighting: ((N-i)/N)^beta favors earlier questions

2. **Turn Efficiency**:
   - If actual turns > target turns: apply penalty of `turn_diff * turn_penalty_weight`
   - If actual turns ≤ target turns: apply reward of `|turn_diff| * turn_reward_weight`

3. **Final Diagnosis**: The last turn receives the diagnosis correctness reward weighted by `diagnosis_reward_weight`

### Budget-Aware Parameters

- `--reward_budget_aware`: Enable budget-aware reward mode (mutually exclusive with `--reward_forward_sim`)
- `--target_num_turns`: Target number of doctor-patient interactions (default: 3)
- `--turn_penalty_weight`: Penalty per turn above target (default: 0.1)
- `--turn_reward_weight`: Reward per turn below target (default: 0.05)
- `--diagnosis_reward_weight`: Weight for final diagnosis correctness (default: 1.0)
- `--temporal_decay_beta`: Decay rate for temporal weighting (default: 1.0)

### Training Output

After training completes, you'll find:
- `{output_dir}/evaluation_results.json` - Final accuracy/metrics on train/test sets
- `{output_dir}/epoch_stats.json` - Per-epoch training statistics including average question utility and turn efficiency
- Model checkpoints and weights

---

## 5. Hyperparameter Sweeps

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

**Note**: Omit `--reward_forward_sim` for faster baseline evaluation. Add it only if you want to compare reward computation methods.

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

### E. Budget-Aware Training Sweep

Sweep different `max_turns` values **with budget-aware training**:

```bash
python3 sweep_max_turns.py \
  --mode train \
  --max_turns_range 2 3 4 5 \
  --base_output_dir outputs/sweep_budget_aware \
  --base_model_name HF_Qwen/Qwen2.5-7B-Instruct \
  --reward_budget_aware \
  --target_num_turns 3 \
  --turn_penalty_weight 0.1 \
  --turn_reward_weight 0.05 \
  --num_train_epochs 3 \
  --learning_rate 1e-5 \
  --max_scenarios 50 \
  --test_size 10
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

## 6. Plotting Sweep Results

After running a sweep, use [plot_sweep.py](plot_sweep.py) to visualize the results.

### Basic Plotting

The script automatically detects whether you ran training or evaluation:

```bash
python3 plot_sweep.py --results_dir outputs/sweep_baseline
```

This will:
1. Print a summary table of all results
2. Create two plots:
   - **Max Turns vs Accuracy**: How accuracy changes with turn limit
   - **Average Turns Taken vs Accuracy**: Efficiency vs performance tradeoff
3. Save the plot to `outputs/sweep_baseline/sweep_eval_results.png`

### Example Output

```
================================================================================
SWEEP RESULTS SUMMARY (EVAL)
================================================================================
Max Turns    Avg Turns    Test Acc     Train Acc
--------------------------------------------------------------------------------
2            2.00         0.450        N/A
3            2.80         0.600        N/A
4            3.50         0.750        N/A
5            4.20         0.800        N/A
================================================================================

Best test accuracy: 0.800 at max_turns=5
Average turns taken: 4.20
================================================================================
```

### Plotting Options

**Specify mode explicitly**:
```bash
python3 plot_sweep.py \
  --results_dir outputs/sweep_fwd_sim \
  --mode train
```

**Custom output path**:
```bash
python3 plot_sweep.py \
  --results_dir outputs/sweep_baseline \
  --save_plot my_custom_plot.png
```

**Skip summary table** (plot only):
```bash
python3 plot_sweep.py \
  --results_dir outputs/sweep_baseline \
  --no_summary
```

### Comparing Baseline vs Trained

To compare baseline and trained models:

```bash
# Plot baseline results
python3 plot_sweep.py \
  --results_dir outputs/sweep_baseline \
  --mode eval \
  --save_plot comparison_baseline.png

# Plot trained results
python3 plot_sweep.py \
  --results_dir outputs/sweep_fwd_sim \
  --mode train \
  --save_plot comparison_trained.png
```

Then compare the two plots side-by-side!

### What the Plots Show

1. **Max Turns vs Accuracy**:
   - X-axis: Maximum allowed turns (2, 3, 4, 5, etc.)
   - Y-axis: Final accuracy on test set
   - Shows how performance improves with more interaction budget

2. **Average Turns Taken vs Accuracy**:
   - X-axis: Average number of turns actually used by the model
   - Y-axis: Final accuracy on test set
   - Shows efficiency: higher accuracy with fewer turns is better
   - Helps identify if the model is using its turn budget wisely

### Plotting After "Both" Mode Sweep

If you ran a sweep with `--mode both`, you can plot either the training or evaluation results:

```bash
# Plot evaluation (baseline) results
python3 plot_sweep.py \
  --results_dir outputs/sweep_both \
  --mode eval \
  --save_plot sweep_both_eval.png

# Plot training results
python3 plot_sweep.py \
  --results_dir outputs/sweep_both \
  --mode train \
  --save_plot sweep_both_train.png
```

---

## 7. Key Parameters Explained

### Model Parameters

- `--base_model_name`: HuggingFace model to use (prefix with `HF_`)
- `--patient_llm`: Model for patient agent (prefix with `HF_` or `VLLM_`, default: same as base)
- `--measurement_llm`: Model for measurement/test agent (prefix with `HF_` or `VLLM_`)
- `--moderator_llm`: Model for diagnosis evaluation (prefix with `HF_` or `VLLM_`)

### vLLM Parameters

- `--use_vllm_policy`: Use vLLM for policy model generation during training and evaluation
- `--vllm_tensor_parallel_size`: Number of GPUs for vLLM tensor parallelism (default: 1)
- `--vllm_gpu_memory_utilization`: GPU memory utilization for vLLM 0.0-1.0 (default: 0.9, reduce to 0.7-0.85 if using `--use_vllm_policy` during training)

### Dataset Parameters

- `--dataset_path`: Path to JSONL dataset (default: `agentclinic_medqa.jsonl`)
- `--max_scenarios`: Limit number of scenarios (useful for quick tests)
- `--test_size`: Number of scenarios for test set (e.g., 10)
- `--test_ratio`: Ratio of scenarios for test set (e.g., 0.2 for 20%)

### Simulation Parameters

- `--max_turns`: Maximum doctor-patient interactions before forced diagnosis
- `--question_cost`: Budget cost per question (default: 1.0)
- `--temperature`: Sampling temperature (0.0 = deterministic)

### Reward Mode Parameters

**Forward Simulation Rewards:**
- `--reward_forward_sim`: Enable forward simulation rewards (per-turn credit assignment via counterfactual simulation)
- `--diagnosis_reward_weight`: Weight for diagnosis correctness (default: 1.0)
- `--intrinsic_token_weight`: Penalty per token generated (default: 0.001)
- `--intrinsic_turn_weight`: Penalty per turn taken (default: 0.0)
- `--forward_sim_temperature`: Temperature for forward simulation (default: 0.0)

**Budget-Aware Rewards:**
- `--reward_budget_aware`: Enable budget-aware rewards (turn efficiency + confidence-based question utility)
- `--target_num_turns`: Target number of doctor-patient interactions (default: 3)
- `--turn_penalty_weight`: Penalty per turn above target (default: 0.1)
- `--turn_reward_weight`: Reward per turn below target (default: 0.05)
- `--diagnosis_reward_weight`: Weight for final diagnosis correctness (default: 1.0)
- `--temporal_decay_beta`: Decay rate for temporal weighting of questions (default: 1.0)

**Note:** `--reward_forward_sim` and `--reward_budget_aware` are mutually exclusive. Use one or the other, not both.

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
# Evaluate baseline on 10 scenarios (fast - no forward simulation)
python3 evaluate.py \
  --max_scenarios 10 \
  --output_dir outputs/quick_baseline
```

### Example 2: Quick Training Test (Forward Simulation)

```bash
# Train with forward sim on 10 scenarios for 1 epoch
python3 ppo.py \
  --max_scenarios 10 \
  --test_size 2 \
  --reward_forward_sim \
  --num_train_epochs 1 \
  --output_dir outputs/quick_train
```

### Example 2b: Quick Training Test (Budget-Aware)

```bash
# Train with budget-aware rewards on 10 scenarios for 1 epoch
python3 ppo.py \
  --max_scenarios 10 \
  --test_size 2 \
  --reward_budget_aware \
  --target_num_turns 3 \
  --num_train_epochs 1 \
  --output_dir outputs/quick_train_budget
```

### Example 3: Full Experiment Comparison

```bash
# Step 1: Baseline evaluation (fast - no forward simulation)
python3 evaluate.py \
  --max_scenarios 50 \
  --test_size 10 \
  --output_dir outputs/exp1_baseline

# Step 2: Train with forward simulation (requires --reward_forward_sim)
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

### Example 4: Comprehensive Sweep with Plotting

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

# Plot baseline results
python3 plot_sweep.py \
  --results_dir outputs/comprehensive_sweep \
  --mode eval \
  --save_plot comprehensive_baseline.png

# Plot training results
python3 plot_sweep.py \
  --results_dir outputs/comprehensive_sweep \
  --mode train \
  --save_plot comprehensive_trained.png
```

---

## Complete Workflow Examples

### Workflow 1: Baseline Sweep + Plot

```bash
# Step 1: Run baseline sweep (fast - no forward simulation)
python3 sweep_max_turns.py \
  --mode eval \
  --max_turns_range 2 3 4 5 6 \
  --max_scenarios 50 \
  --test_size 10 \
  --base_output_dir outputs/baseline_sweep

# Step 2: Plot results
python3 plot_sweep.py \
  --results_dir outputs/baseline_sweep
# Output: outputs/baseline_sweep/sweep_eval_results.png
```

### Workflow 2: Training Sweep + Plot

```bash
# Step 1: Run training sweep with forward simulation
python3 sweep_max_turns.py \
  --mode train \
  --max_turns_range 3 4 5 \
  --max_scenarios 50 \
  --test_size 10 \
  --reward_forward_sim \
  --intrinsic_token_weight 0.001 \
  --num_train_epochs 3 \
  --use_lora \
  --base_output_dir outputs/training_sweep

# Step 2: Plot results
python3 plot_sweep.py \
  --results_dir outputs/training_sweep \
  --mode train
# Output: outputs/training_sweep/sweep_train_results.png
```

### Workflow 3: Compare Baseline vs Trained

```bash
# Step 1: Baseline sweep (fast - no forward simulation)
python3 sweep_max_turns.py \
  --mode eval \
  --max_turns_range 3 4 5 \
  --max_scenarios 100 \
  --test_size 20 \
  --base_output_dir outputs/comparison/baseline

# Step 2: Training sweep (requires --reward_forward_sim)
python3 sweep_max_turns.py \
  --mode train \
  --max_turns_range 3 4 5 \
  --max_scenarios 100 \
  --test_size 20 \
  --reward_forward_sim \
  --num_train_epochs 5 \
  --use_lora \
  --base_output_dir outputs/comparison/trained

# Step 3: Plot both
python3 plot_sweep.py \
  --results_dir outputs/comparison/baseline \
  --save_plot comparison_baseline.png

python3 plot_sweep.py \
  --results_dir outputs/comparison/trained \
  --mode train \
  --save_plot comparison_trained.png

# Now you can compare comparison_baseline.png vs comparison_trained.png!
```

### Workflow 4: Compare Forward Simulation vs Budget-Aware Training

```bash
# Step 1: Baseline sweep
python3 sweep_max_turns.py \
  --mode eval \
  --max_turns_range 3 4 5 \
  --max_scenarios 100 \
  --test_size 20 \
  --base_output_dir outputs/comparison/baseline

# Step 2: Forward simulation training
python3 sweep_max_turns.py \
  --mode train \
  --max_turns_range 3 4 5 \
  --max_scenarios 100 \
  --test_size 20 \
  --reward_forward_sim \
  --intrinsic_token_weight 0.001 \
  --num_train_epochs 5 \
  --use_lora \
  --base_output_dir outputs/comparison/forward_sim

# Step 3: Budget-aware training
python3 sweep_max_turns.py \
  --mode train \
  --max_turns_range 3 4 5 \
  --max_scenarios 100 \
  --test_size 20 \
  --reward_budget_aware \
  --target_num_turns 3 \
  --turn_penalty_weight 0.1 \
  --turn_reward_weight 0.05 \
  --num_train_epochs 5 \
  --use_lora \
  --base_output_dir outputs/comparison/budget_aware

# Step 4: Plot all three
python3 plot_sweep.py \
  --results_dir outputs/comparison/baseline \
  --save_plot comparison_baseline.png

python3 plot_sweep.py \
  --results_dir outputs/comparison/forward_sim \
  --mode train \
  --save_plot comparison_forward_sim.png

python3 plot_sweep.py \
  --results_dir outputs/comparison/budget_aware \
  --mode train \
  --save_plot comparison_budget_aware.png

# Now you can compare all three reward approaches!
```

---

## Notes

1. **Reward mode requirements**:
   - **Training (`ppo.py`)**: REQUIRED - you must use either `--reward_forward_sim` or `--reward_budget_aware`
   - **Baseline evaluation (`evaluate.py`)**: OPTIONAL - omit for faster evaluation, or include to compare reward signals
   - The two reward modes are mutually exclusive - use one or the other, not both
   - Older reward modes (sparse, dense, correctness_baseline) have been archived

2. **Reward mode comparison**:
   - **Forward Simulation** (`--reward_forward_sim`): Computes per-turn rewards via counterfactual "what-if" simulations
   - **Budget-Aware** (`--reward_budget_aware`): Optimizes for turn efficiency with a target number of turns
   - Both support LoRA and 4-bit quantization for memory efficiency

3. **Baseline vs Training**:
   - Use `evaluate.py` for baseline (no training) - fast correctness-only evaluation
   - Use `ppo.py` for training with your chosen reward mode

4. **Memory considerations**:
   - For large models, use `--use_lora --use_4bit`
   - Consider `--disable_reference_model` if memory is tight
   - Budget-aware rewards may be more memory-efficient than forward simulation as they compute confidence directly rather than running full counterfactual simulations

5. **vLLM acceleration**:
   - Use `VLLM_` prefix for patient, measurement, and moderator agents for 2-5x faster inference
   - Use `--use_vllm_policy` for policy model during both training and evaluation
   - During training: vLLM handles generation; HuggingFace model handles gradients/PPO updates
   - vLLM requires separate installation: `pip install vllm`
   - Gracefully falls back to HuggingFace if vLLM is not installed
   - Supports multi-GPU tensor parallelism via `--vllm_tensor_parallel_size`
   - When using `--use_vllm_policy` during training, reduce `--vllm_gpu_memory_utilization` to 0.7-0.85 to avoid OOM

6. **Reproducibility**: Always set `--seed 42` for reproducible results

7. **Sweep organization**: Each sweep run creates a descriptive directory name based on hyperparameters (e.g., `train_mt5_lr1e-05_ep3_fwd_sim_lora` or `train_mt5_lr1e-05_ep3_budget_aware`)
