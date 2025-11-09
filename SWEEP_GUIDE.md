# Max Turns Sweep Guide

This guide explains how to run parameter sweeps over `max_turns` and analyze the results.

## Quick Start

### 1. Run Baseline Sweep (No Training)

Evaluate the base model with different max_turns settings:

```bash
python sweep_max_turns.py \
    --mode eval \
    --max_turns_range 2 3 4 5 \
    --base_output_dir outputs/baseline_sweep \
    --test_size 20 \
    --use_4bit
```

### 2. Run Training Sweep

Train models with different max_turns settings:

```bash
python sweep_max_turns.py \
    --mode train \
    --max_turns_range 2 3 4 5 \
    --base_output_dir outputs/training_sweep \
    --test_size 20 \
    --num_train_epochs 50 \
    --learning_rate 1e-5 \
    --use_4bit \
    --use_lora \
    --disable_reference_model \
    --reward_correctness_baseline
```

### 3. Plot Results

```bash
# For baseline sweep
python plot_sweep.py --results_dir outputs/baseline_sweep

# For training sweep
python plot_sweep.py --results_dir outputs/training_sweep
```

## Output Structure

After running a sweep, you'll have:

```
outputs/sweep_max_turns/
├── sweep_results.json                    # Combined results
├── train_sweep_results.json              # Training results only
├── eval_sweep_results.json               # Evaluation results only
├── sweep_train_results.png              # Automatically saved plots
│
├── train_mt2_lr1e-05_ep50_corr_only_lora_4bit/   # Run for max_turns=2
│   ├── epoch_stats.json                  # Training progress
│   ├── evaluation_results.json           # Final test results
│   ├── adapter_config.json               # Model checkpoints
│   └── ...
│
├── train_mt3_lr1e-05_ep50_corr_only_lora_4bit/   # Run for max_turns=3
│   └── ...
│
└── ...
```

## Informative Run Names

The sweep script automatically creates informative directory names based on hyperparameters:

**Format:** `{mode}_mt{max_turns}_lr{lr}_ep{epochs}_{reward_config}_{model_config}`

**Examples:**
- `train_mt3_lr1e-05_ep50_corr_only_lora_4bit`
  - Training mode
  - max_turns=3
  - learning_rate=1e-05
  - 50 epochs
  - Correctness-only reward
  - LoRA enabled
  - 4-bit quantization

- `eval_mt5_4bit`
  - Evaluation mode
  - max_turns=5
  - 4-bit quantization

- `train_mt4_lr1e-05_ep50_fwd_sim_lora_4bit`
  - Training mode
  - max_turns=4
  - learning_rate=1e-05
  - 50 epochs
  - Forward simulation reward (with default intrinsic weights)
  - LoRA enabled
  - 4-bit quantization

## Plots Generated

The plotting script creates two side-by-side plots:

1. **Max Turns (Limit) vs Accuracy**
   - X-axis: Maximum allowed turns (budget constraint)
   - Y-axis: Test accuracy
   - Shows how much budget the model needs

2. **Average Turns Taken vs Accuracy**
   - X-axis: Average turns actually used by the model
   - Y-axis: Test accuracy
   - Shows model efficiency

## Advanced Options

### Custom Hyperparameters

```bash
python sweep_max_turns.py \
    --mode train \
    --max_turns_range 2 3 4 5 6 \
    --learning_rate 5e-6 \
    --num_train_epochs 100 \
    --diagnosis_reward_weight 2.0 \
    --budget_reward_weight 0.5 \
    --question_reward_weight 1.0 \
    --test_size 50
```

### Reward Modes

**Sparse Reward (Standard RL)**:
Use `--reward_sparse` for standard RL approach where only the diagnosis turn gets substantial reward:
- Question turns: `-question_cost` (small penalty encouraging efficiency)
- Diagnosis turn: `correctness + budget_bonus`

This is more aligned with RL literature and lets PPO's value function handle credit assignment automatically.

```bash
python sweep_max_turns.py \
    --mode train \
    --max_turns_range 2 3 4 5 \
    --reward_sparse \
    --question_cost 0.1 \
    --test_size 20
```

**Correctness-Only Baseline**:
Use `--reward_correctness_baseline` for simplest baseline (only diagnosis correctness):

```bash
python sweep_max_turns.py \
    --mode train \
    --max_turns_range 2 3 4 5 \
    --reward_correctness_baseline \
    --test_size 20
```

**Dense Reward (Default)**:
Without either flag, uses dense reward shaping with:
- Question utility (temporally weighted)
- Budget efficiency
- Diagnosis correctness

**Forward Simulation Reward**:
Use `--reward_forward_sim` for per-turn credit assignment via forward simulation:
- For each doctor turn: forward simulate to episode end and compute extrinsic reward (diagnosis correctness)
- Add intrinsic rewards: token count penalty + turn penalty
- Combined: `r_t = diagnosis_weight * extrinsic_correctness_t + intrinsic_t`
- Note: This is 3-4x slower due to O(N²) forward simulations

```bash
python sweep_max_turns.py \
    --mode train \
    --max_turns_range 2 3 4 5 \
    --reward_forward_sim \
    --intrinsic_token_weight 0.001 \
    --intrinsic_turn_weight 0.0 \
    --forward_sim_temperature 0.0 \
    --test_size 20
```

### Continue on Error

If one run fails, continue with the rest:

```bash
python sweep_max_turns.py \
    --mode train \
    --max_turns_range 2 3 4 5 \
    --continue_on_error
```

### Different Base Models

```bash
python sweep_max_turns.py \
    --mode eval \
    --max_turns_range 2 3 4 5 \
    --base_model_name HF_meta-llama/Llama-3.2-7B-Instruct \
    --patient_llm HF_Qwen/Qwen2.5-7B-Instruct \
    --measurement_llm HF_Qwen/Qwen2.5-7B-Instruct
```

## Understanding Results

### Summary Table

The plot script prints a summary:

```
================================================================================
SWEEP RESULTS SUMMARY (EVAL)
================================================================================
Max Turns    Avg Turns    Test Acc     Train Acc
--------------------------------------------------------------------------------
2            2.00         0.450        N/A
3            2.85         0.550        N/A
4            3.20         0.650        N/A
5            3.45         0.700        N/A
================================================================================

Best test accuracy: 0.700 at max_turns=5
Average turns taken: 3.45
================================================================================
```

### Key Insights

From the plots, you can answer:
1. **What's the minimum budget needed?** - Where accuracy plateaus on the left plot
2. **Is the model efficient?** - Compare actual turns used (right plot) vs budget (left plot)
3. **Diminishing returns?** - Where accuracy stops improving despite more budget

## Comparison Workflow

Compare baseline vs trained models:

```bash
# 1. Baseline sweep
python sweep_max_turns.py \
    --mode eval \
    --max_turns_range 2 3 4 5 \
    --base_output_dir outputs/baseline \
    --test_size 20

# 2. Training sweep
python sweep_max_turns.py \
    --mode train \
    --max_turns_range 2 3 4 5 \
    --base_output_dir outputs/trained \
    --test_size 20 \
    --num_train_epochs 50

# 3. Plot both
python plot_sweep.py --results_dir outputs/baseline
python plot_sweep.py --results_dir outputs/trained

# 4. Compare side by side
# (Open both PNGs for visual comparison)
```

## Tips

1. **Start small**: Test with `--max_scenarios 10` first
2. **Save compute**: Use `--use_4bit` for faster experiments
3. **Test range**: Try `--max_turns_range 2 3 4 5 6 7 8` to find optimal budget
4. **Check results early**: Each run saves `evaluation_results.json` immediately

## Troubleshooting

**Problem**: Sweep fails partway through
- **Solution**: Use `--continue_on_error` to skip failed runs

**Problem**: Out of memory
- **Solution**: Add `--use_4bit` or reduce `--num_train_epochs`

**Problem**: Results look wrong
- **Solution**: Check that `--test_size` and `--seed` are consistent across sweeps

**Problem**: Plot shows no test accuracy
- **Solution**: Make sure you specified `--test_size` or `--test_ratio` in the sweep
