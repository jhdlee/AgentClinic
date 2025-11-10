"""
Utility functions for PPO training and evaluation.
"""

import os
from typing import Optional, List
from collections import deque


def _stringify_context(value) -> str:
    """Recursively stringify nested dicts/lists for context display."""
    if isinstance(value, dict):
        return "; ".join(f"{k}: {_stringify_context(v)}" for k, v in value.items())
    if isinstance(value, list):
        return "; ".join(_stringify_context(v) for v in value)
    return str(value)


class CheckpointManager:
    """Manages checkpoint rotation to keep only N most recent checkpoints."""

    def __init__(self, output_dir: str, save_total_limit: Optional[int]):
        self.output_dir = output_dir
        self.save_total_limit = save_total_limit
        self.checkpoints: deque = deque(maxlen=save_total_limit if save_total_limit else None)

    def add_checkpoint(self, checkpoint_dir: str):
        """Add a new checkpoint and remove oldest if limit exceeded."""
        if self.save_total_limit is None or self.save_total_limit <= 0:
            return

        self.checkpoints.append(checkpoint_dir)

        # If we exceeded the limit, remove the oldest
        if len(self.checkpoints) > self.save_total_limit:
            oldest = self.checkpoints.popleft()
            if os.path.exists(oldest) and oldest != checkpoint_dir:
                import shutil
                shutil.rmtree(oldest, ignore_errors=True)


def save_checkpoint(trainer, tokenizer, output_dir: str, checkpoint_name: str, checkpoint_manager: CheckpointManager):
    """Save a training checkpoint and manage rotation."""
    checkpoint_dir = os.path.join(output_dir, checkpoint_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    trainer.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    checkpoint_manager.add_checkpoint(checkpoint_dir)

    return checkpoint_dir


def verify_ppo_gradients(trainer, stats, context: str = ""):
    """
    Verify PPO gradient health.

    Checks for common gradient issues like NaN, inf, or vanishing gradients.
    """
    import torch

    model = trainer.model
    has_nan = False
    has_inf = False
    max_grad = 0.0
    min_grad = float('inf')

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if torch.isnan(param.grad).any():
                has_nan = True
                print(f"[GRADIENT CHECK] NaN gradient in {name} ({context})")
            if torch.isinf(param.grad).any():
                has_inf = True
                print(f"[GRADIENT CHECK] Inf gradient in {name} ({context})")
            max_grad = max(max_grad, grad_norm)
            if grad_norm > 0:
                min_grad = min(min_grad, grad_norm)

    if has_nan or has_inf:
        print(f"[GRADIENT CHECK] CRITICAL: Gradient issues detected! ({context})")

    if max_grad > 100.0:
        print(f"[GRADIENT CHECK] WARNING: Large gradient detected: {max_grad:.2f} ({context})")

    if min_grad < 1e-8 and min_grad != float('inf'):
        print(f"[GRADIENT CHECK] WARNING: Very small gradient: {min_grad:.2e} ({context})")

    print(f"[GRADIENT CHECK] Gradient range: [{min_grad:.2e}, {max_grad:.2e}] ({context})")
