"""
Archive of alternative reward computation modes.

This file contains reward modes that were explored but are not currently in use.
The main implementation now uses only forward simulation rewards.

Kept for reference and potential future experiments.
"""

from typing import Dict, List, Tuple, Any


def compute_sparse_reward(
    state,
    correctness: float,
    question_cost: float,
    diagnosis_reward_weight: float,
    budget_reward_weight: float,
    debug_print: bool = False,
    reward_breakdown_debug: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """
    Sparse reward: only diagnosis turn gets substantial reward, questions get small cost.

    This is standard RL approach where:
    - Question turns: -question_cost (small penalty encouraging efficiency)
    - Diagnosis turn: correctness + budget_bonus

    Lets PPO's value function handle credit assignment automatically.
    """
    budget_fraction = state.remaining_budget / float(max(state.max_turns, 1))
    diagnosis_component = diagnosis_reward_weight * correctness
    budget_component = budget_reward_weight * budget_fraction
    diagnosis_reward = diagnosis_component + budget_component

    per_turn_rewards = []
    per_turn_components = []

    for turn in state.turns:
        if turn.action.type == "diagnosis":
            # Diagnosis turn gets full reward
            per_turn_rewards.append(diagnosis_reward)
            per_turn_components.append({
                "diagnosis": diagnosis_component,
                "budget": budget_component,
                "question": 0.0,
            })
        else:
            # Question turns get small cost (negative reward encourages efficiency)
            question_penalty = -question_cost
            per_turn_rewards.append(question_penalty)
            per_turn_components.append({
                "diagnosis": 0.0,
                "budget": 0.0,
                "question": question_penalty,
            })

    reward = sum(per_turn_rewards)

    if debug_print or reward_breakdown_debug:
        print(
            f"[Episode {state.scenario_id}] Sparse reward | "
            f"correctness={correctness:.3f} (w={diagnosis_reward_weight}) | "
            f"budget={budget_fraction:.3f} (w={budget_reward_weight}) | "
            f"question_cost={question_cost:.3f} | total={reward:.3f}"
        )
        if per_turn_components:
            print(f"[Episode {state.scenario_id}] Per-turn reward breakdown (sparse):")
            for idx, (turn, components, total) in enumerate(
                zip(state.turns, per_turn_components, per_turn_rewards), start=1
            ):
                print(
                    "  Turn {turn_idx} ({action}): total={total:.3f} | "
                    "diagnosis={diag:.3f} | budget={budget:.3f} | question={q:.3f}".format(
                        turn_idx=idx,
                        action=turn.action.type,
                        total=total,
                        diag=components["diagnosis"],
                        budget=components["budget"],
                        q=components["question"],
                    )
                )

    return reward, {
        "correctness": correctness,
        "budget_saved": budget_fraction,
        "question_reward": sum(r for r in per_turn_rewards if r < 0),
        "reward_per_turn": per_turn_rewards,
        "reward_breakdown_per_turn": per_turn_components,
    }


def compute_correctness_baseline_reward(
    state,
    correctness: float,
    diagnosis_reward_weight: float,
    debug_print: bool = False,
    reward_breakdown_debug: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """
    Correctness-only baseline: simplest reward (only diagnosis correctness for every turn).
    """
    diagnosis_component = diagnosis_reward_weight * correctness
    per_turn_components = [
        {"diagnosis": diagnosis_component, "budget": 0.0, "question": 0.0}
        for _ in state.turns
    ]
    per_turn_rewards = [diagnosis_component for _ in state.turns]
    reward = diagnosis_component

    if debug_print or reward_breakdown_debug:
        print(
            f"[Episode {state.scenario_id}] Correctness baseline | "
            f"correctness={correctness:.3f} (w={diagnosis_reward_weight}) | "
            f"reward={reward:.3f}"
        )

    return reward, {
        "correctness": correctness,
        "budget_saved": 0.0,
        "question_reward": 0.0,
        "reward_per_turn": per_turn_rewards,
        "reward_breakdown_per_turn": per_turn_components,
    }


def compute_dense_reward(
    state,
    correctness: float,
    generate_fn,
    question_cost: float,
    temporal_decay_beta: float,
    budget_reward_weight: float,
    question_reward_weight: float,
    diagnosis_reward_weight: float,
    question_confidence_gain_fn,
    debug_print: bool = False,
    reward_breakdown_debug: bool = False,
    utility_warning_emitted: bool = False,
) -> Tuple[float, Dict[str, Any], bool]:
    """
    Dense reward shaping with:
    - Question utility (temporally weighted)
    - Budget efficiency
    - Diagnosis correctness

    Returns:
        reward: Total episode reward
        info: Dict with reward components
        utility_warning_emitted: Updated warning flag
    """
    question_reward_total = 0.0
    total_questions = sum(1 for action in state.actions if action.type != "diagnosis")
    per_turn_components: List[Dict[str, float]] = [
        {"diagnosis": 0.0, "budget": 0.0, "question": 0.0} for _ in state.turns
    ]

    if total_questions > 0 and question_reward_weight != 0.0:
        question_counter = 0
        for idx, turn in enumerate(state.turns):
            action = turn.action
            if action.type == "diagnosis":
                continue
            # Use counter BEFORE incrementing (0-indexed: first question gets i=0)
            base = max((total_questions - question_counter) / total_questions, 0.0)
            temporal_weight = base ** temporal_decay_beta
            try:
                if debug_print:
                    print(f"Computing question confidence gain for turn {idx + 1} of {total_questions}")
                utility = question_confidence_gain_fn(
                    state=state,
                    turn=turn,
                    generate_fn=generate_fn,
                )
            except Exception as e:
                print(f"Warning: _question_confidence_gain raised {e}")
                utility = 0.0
                if not utility_warning_emitted:
                    print(
                        "Warning: Failed to compute question confidence gain; setting to 0. "
                        "This may occur if the moderator backend is unavailable or returns invalid data."
                    )
                    utility_warning_emitted = True
            question_component = temporal_weight * utility
            question_reward_total += question_component
            if per_turn_components:
                per_turn_components[idx]["question"] += (
                    question_reward_weight * question_component
                )
            # Increment counter AFTER computing weight (moves to next 0-indexed position)
            question_counter += 1

    budget_fraction = state.remaining_budget / float(max(state.max_turns, 1))
    diagnosis_component = diagnosis_reward_weight * correctness
    budget_component = budget_reward_weight * budget_fraction
    question_component_weighted = question_reward_weight * question_reward_total

    reward = diagnosis_component + budget_component + question_component_weighted

    # Distribute total reward equally across all turns for PPO
    per_turn_rewards = [reward for _ in state.turns]

    # But also apply diagnosis reward only to the final (diagnosis) turn's component
    if state.turns:
        per_turn_components[-1]["diagnosis"] = diagnosis_component
        per_turn_components[-1]["budget"] = budget_component

    if debug_print or reward_breakdown_debug:
        print(
            f"[Episode {state.scenario_id}] Dense reward | "
            f"correctness={correctness:.3f} (w={diagnosis_reward_weight}) | "
            f"budget={budget_fraction:.3f} (w={budget_reward_weight}) | "
            f"question_sum={question_reward_total:.3f} (w={question_reward_weight}) | "
            f"total={reward:.3f}"
        )
        if per_turn_components:
            print(f"[Episode {state.scenario_id}] Per-turn component breakdown (dense):")
            for idx, (turn, components) in enumerate(zip(state.turns, per_turn_components), start=1):
                print(
                    "  Turn {turn_idx} ({action}): diagnosis={diag:.3f} | "
                    "budget={budget:.3f} | question={q:.3f}".format(
                        turn_idx=idx,
                        action=turn.action.type,
                        diag=components["diagnosis"],
                        budget=components["budget"],
                        q=components["question"],
                    )
                )

    return reward, {
        "correctness": correctness,
        "budget_saved": budget_fraction,
        "question_reward": question_component_weighted,
        "reward_per_turn": per_turn_rewards,
        "reward_breakdown_per_turn": per_turn_components,
    }, utility_warning_emitted
