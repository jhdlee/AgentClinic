#!/usr/bin/env python3
"""PPO fine-tuning loop for the AgentClinic doctor model.

This script adapts the multiturn PPO setup from CollabLLM to the AgentClinic
simulated doctor–patient environment. It handles scenario loading, on-policy
rollouts against a lightweight patient simulator, counterfactual question
analysis, and PPO optimisation of a causal LLM with optional LoRA and
4-bit quantisation.

Key reward design (doctor-centric):
* Diagnosis term (weightable) based on moderator-verified correctness.
* Budget term (weightable) based on unused interaction budget.
* Question utility term (weightable) obtained from counterfactual rollouts that
  forbid re-asking the question and measure the drop in final correctness. The
  contribution of each question is temporally re-weighted by
  `((N - i) / N) ** β`.

The script assumes vLLM is unavailable and generates completions directly from
the policy model, using `torch.no_grad()` during rollouts to avoid tracking
gradients while keeping the model in training mode.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Suppress vLLM progress bars by setting environment variables before vLLM import
os.environ['VLLM_CONFIGURE_LOGGING'] = '0'
os.environ['VLLM_LOGGING_LEVEL'] = 'WARNING'

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

try:  # Optional dependency for 4-bit loading
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - optional dependency
    BitsAndBytesConfig = None  # type: ignore

try:  # Optional dependency for vLLM
    from vllm import LLM, SamplingParams
except ImportError:  # pragma: no cover - optional dependency
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore

from transformers import AutoTokenizer, set_seed, pipeline

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)

# Import utility functions
from utils import _stringify_context, CheckpointManager, save_checkpoint, verify_ppo_gradients


# ---------------------------------------------------------------------------
# HuggingFace utility functions
# ---------------------------------------------------------------------------

def load_huggingface_model(model):
    pipe = pipeline("text-generation", model=model, device_map="auto")
    return pipe

def inference_huggingface(prompt, pipe, max_new_tokens=200, temperature=0.0):
    """
    Run inference on a Hugging Face pipeline.
    
    Args:
        prompt: The formatted input text
        pipe: The Hugging Face pipeline
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature (lower = more deterministic)
    
    Returns:
        The generated text (with prompt removed)
    """
    # Generate with parameters matching your other models
    response = pipe(
        prompt, 
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,  # Use sampling if temperature > 0
        return_full_text=False  # This removes the prompt automatically
    )[0]["generated_text"]
    
    # Clean up any leading/trailing whitespace
    response = response.strip()
    
    return response

# Simple cache for HF pipelines so we only load once per model id
HUGGINGFACE_PIPES = {}

# ---------------------------------------------------------------------------
# vLLM utility functions
# ---------------------------------------------------------------------------

def suppress_vllm_logging():
    """
    Suppress vLLM's verbose INFO-level logging.
    Sets vLLM loggers to WARNING level to reduce noise.
    """
    vllm_loggers = [
        "vllm.engine.llm_engine",
        "vllm.engine.async_llm_engine",
        "vllm.executor.gpu_executor",
        "vllm.worker.worker",
        "vllm.config",
        "vllm.model_executor.model_loader",
        "vllm",
    ]
    for logger_name in vllm_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

def load_vllm_model(model, tensor_parallel_size=1, gpu_memory_utilization=0.9, verbose=False):
    """
    Load a vLLM model for faster inference.

    Args:
        model: The model name/path
        tensor_parallel_size: Number of GPUs to use for tensor parallelism
        gpu_memory_utilization: GPU memory utilization (0.0 to 1.0)
        verbose: If False (default), suppress vLLM's verbose logging

    Returns:
        The vLLM LLM instance
    """
    if LLM is None:
        raise ImportError("vLLM is not installed. Install it with: pip install vllm")

    # Suppress vLLM logging unless verbose mode is enabled
    if not verbose:
        suppress_vllm_logging()

    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        disable_log_stats=not verbose,  # Disable progress bars unless verbose mode
    )
    return llm

def inference_vllm(prompt, llm, max_new_tokens=200, temperature=0.0):
    """
    Run inference on a vLLM model.

    Args:
        prompt: The formatted input text
        llm: The vLLM LLM instance
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature (lower = more deterministic)

    Returns:
        The generated text (with prompt removed)
    """
    if SamplingParams is None:
        raise ImportError("vLLM is not installed. Install it with: pip install vllm")

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        top_p=1.0,
    )

    outputs = llm.generate([prompt], sampling_params)
    response = outputs[0].outputs[0].text.strip()

    return response

# Simple cache for vLLM models so we only load once per model id
VLLM_MODELS = {}

def query_model(model, prompt, system_prompt, tries=30, timeout=20.0, max_prompt_len=2**14, clip_prompt=False, vllm_tensor_parallel_size=1, vllm_gpu_memory_utilization=0.9, vllm_verbose=False):
    if clip_prompt:
        prompt = prompt[:max_prompt_len]

    # Load vLLM model once outside retry loop (if applicable)
    if isinstance(model, str) and model.startswith("VLLM_"):
        model_id = model[5:]

        # Load or retrieve cached vLLM model
        llm = VLLM_MODELS.get(model_id)
        if llm is None:
            logger.info("Loading vLLM model: %s", model_id)
            llm = load_vllm_model(
                model_id,
                tensor_parallel_size=vllm_tensor_parallel_size,
                gpu_memory_utilization=vllm_gpu_memory_utilization,
                verbose=vllm_verbose
            )
            VLLM_MODELS[model_id] = llm
            logger.info("vLLM model cached: %s", model_id)

        # Retry only the inference part
        for _ in range(tries):
            try:
                input_text = f"{system_prompt}\n\n{prompt}"
                answer = inference_vllm(input_text, llm)
                answer = re.sub(r"\s+", " ", answer)
                return answer
            except Exception as e:
                logger.warning("vLLM inference failed, retrying: %s", str(e))
                time.sleep(timeout)
                continue
        raise Exception("Max retries: timeout")

    # For HuggingFace models, keep original retry logic
    for _ in range(tries):
        try:
            if isinstance(model, str):
                if model.startswith("HF_"):
                    # Extract the HF repo id
                    hf_id = model[3:]

                    # Load or retrieve cached pipeline
                    pipe = HUGGINGFACE_PIPES.get(hf_id)
                    if pipe is None:
                        pipe = load_huggingface_model(hf_id)
                        HUGGINGFACE_PIPES[hf_id] = pipe

                    # Format the prompt appropriately for instruction-tuned models
                    # Many HF models use chat templates
                    if hasattr(pipe.tokenizer, 'apply_chat_template') and pipe.tokenizer.chat_template:
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ]
                        input_text = pipe.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    else:
                        # Fallback for models without chat templates
                        input_text = f"{system_prompt}\n\n{prompt}"

                    answer = inference_huggingface(input_text, pipe)
                    answer = re.sub(r"\s+", " ", answer)

                    return answer
                else:
                    raise ValueError("Model backends must be prefixed with 'HF_' (HuggingFace) or 'VLLM_' (vLLM).")
            else:
                # Assume it's a model object/path for HuggingFace
                pipe = load_huggingface_model(model)

                # Format the prompt appropriately for instruction-tuned models
                # Many HF models use chat templates
                if hasattr(pipe.tokenizer, 'apply_chat_template') and pipe.tokenizer.chat_template:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                    input_text = pipe.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    # Fallback for models without chat templates
                    input_text = f"{system_prompt}\n\n{prompt}"

                answer = inference_huggingface(input_text, pipe)
                answer = re.sub(r"\s+", " ", answer)

                return answer

        except Exception:
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class ScenarioMedQA:
    def __init__(self, scenario_dict, scenario_id: int) -> None:
        self.scenario_id = scenario_id
        self.scenario_dict = scenario_dict
        exam = scenario_dict["OSCE_Examination"]
        self.tests = exam.get("Test_Results", {})
        self.diagnosis = exam.get("Correct_Diagnosis", "")
        self.patient_info = exam.get("Patient_Actor", {})
        self.examiner_info = exam.get("Objective_for_Doctor", "")
        self.physical_exams = exam.get("Physical_Examination_Findings", {})

    def patient_information(self):
        return self.patient_info

    def examiner_information(self):
        return self.examiner_info

    def exam_information(self):
        exams = dict(self.physical_exams)
        if "tests" not in exams:
            exams["tests"] = self.tests
        return exams

    def diagnosis_information(self) -> str:
        return self.diagnosis


class ScenarioLoaderMedQA:
    def __init__(
        self,
        path: str = "agentclinic_medqa.jsonl",
        max_scenarios: Optional[int] = None,
        test_size: Optional[int] = None,
        test_ratio: Optional[float] = None,
        seed: int = 42,
    ) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            scenario_strs = [json.loads(line) for line in f if line.strip()]

        if max_scenarios is not None:
            scenario_strs = scenario_strs[:max_scenarios]

        self.scenarios = [ScenarioMedQA(data, idx) for idx, data in enumerate(scenario_strs)]
        self.num_scenarios = len(self.scenarios)

        # Split into train/test if requested
        self.train_indices = list(range(self.num_scenarios))
        self.test_indices = []

        if test_size is not None or test_ratio is not None:
            rng = random.Random(seed)
            all_indices = list(range(self.num_scenarios))
            rng.shuffle(all_indices)

            if test_size is not None:
                num_test = min(test_size, self.num_scenarios)
            else:
                num_test = int(self.num_scenarios * test_ratio)

            self.test_indices = sorted(all_indices[:num_test])
            self.train_indices = sorted(all_indices[num_test:])

            logger.info(
                "Split scenarios: %d train, %d test (total: %d)",
                len(self.train_indices),
                len(self.test_indices),
                self.num_scenarios,
            )

    def sample_scenario(self):
        return random.choice(self.scenarios)

    def get_scenario(self, id):
        if id is None:
            return self.sample_scenario()
        return self.scenarios[id]

    def get_train_indices(self):
        return self.train_indices

    def get_test_indices(self):
        return self.test_indices

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
class PatientAgent:
    def __init__(self, scenario, backend_str="gpt4", vllm_tensor_parallel_size=1, vllm_gpu_memory_utilization=0.9, vllm_verbose=False) -> None:
        # disease of patient, or "correct answer"
        self.disease = ""
        # symptoms that patient presents
        self.symptoms = ""
        # conversation history between doctor and patient
        self.agent_hist = ""
        # language model backend for patient agent
        self.backend = backend_str
        # sample initial question from dataset
        self.scenario = scenario
        self.reset()
        self.pipe = None
        # vLLM configuration
        self.vllm_tensor_parallel_size = vllm_tensor_parallel_size
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self.vllm_verbose = vllm_verbose

    def inference_patient(self, question) -> str:
        answer = query_model(
            self.backend,
            "\nHere is a history of your dialogue: " + self.agent_hist + "\n Here was the doctor response: " + question + "Now please continue your dialogue\nPatient: ",
            self.system_prompt(),
            vllm_tensor_parallel_size=self.vllm_tensor_parallel_size,
            vllm_gpu_memory_utilization=self.vllm_gpu_memory_utilization,
            vllm_verbose=self.vllm_verbose
        )
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def system_prompt(self) -> str:
        base = """You are a patient in a clinic who only responds in the form of dialogue. You are being inspected by a doctor who will ask you questions and will perform exams on you in order to understand your disease. Your answer will only be 1-3 sentences in length."""
        symptoms = "\n\nBelow is all of your information. {}. \n\n Remember, you must not reveal your disease explicitly but may only convey the symptoms you have in the form of dialogue if you are asked.".format(self.symptoms)
        return base + symptoms
    
    def reset(self) -> None:
        self.agent_hist = ""
        self.symptoms = self.scenario.patient_information()

    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"

class MeasurementAgent:
    def __init__(self, scenario, backend_str="gpt4", vllm_tensor_parallel_size=1, vllm_gpu_memory_utilization=0.9, vllm_verbose=False) -> None:
        # conversation history between doctor and patient
        self.agent_hist = ""
        # presentation information for measurement
        self.presentation = ""
        # language model backend for measurement agent
        self.backend = backend_str
        # prepare initial conditions for LLM
        self.scenario = scenario
        self.pipe = None
        self.reset()
        # vLLM configuration
        self.vllm_tensor_parallel_size = vllm_tensor_parallel_size
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self.vllm_verbose = vllm_verbose

    def inference_measurement(self, question) -> str:
        answer = str()
        answer = query_model(
            self.backend,
            "\nHere is a history of the dialogue: " + self.agent_hist + "\n Here was the doctor measurement request: " + question,
            self.system_prompt(),
            vllm_tensor_parallel_size=self.vllm_tensor_parallel_size,
            vllm_gpu_memory_utilization=self.vllm_gpu_memory_utilization,
            vllm_verbose=self.vllm_verbose
        )
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def system_prompt(self) -> str:
        base = "You are an measurement reader who responds with medical test results. Please respond in the format \"RESULTS: [results here]\""
        presentation = "\n\nBelow is all of the information you have. {}. \n\n If the requested results are not in your data then you can respond with NORMAL READINGS.".format(self.information)
        return base + presentation
    
    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"

    def reset(self) -> None:
        self.agent_hist = ""
        self.information = self.scenario.exam_information()


def compare_results(diagnosis, correct_diagnosis, moderator_llm, vllm_tensor_parallel_size=1, vllm_gpu_memory_utilization=0.9, vllm_verbose=False):
    answer = query_model(
        moderator_llm,
        "\nHere is the correct diagnosis: " + correct_diagnosis + "\n Here was the doctor dialogue: " + diagnosis + "\nAre these the same?",
        "You are responsible for determining if the corrent diagnosis and the doctor diagnosis are the same disease. Please respond only with Yes or No. Nothing else.",
        vllm_tensor_parallel_size=vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_verbose=vllm_verbose
    )
    return answer.lower()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class DoctorEpisodeState:
    """Mutable state for a single doctor–patient episode."""

    scenario_id: int
    scenario: ScenarioMedQA
    remaining_budget: float
    max_turns: int
    history: List[Dict[str, str]] = field(default_factory=list)
    actions: List[DoctorAction] = field(default_factory=list)
    done: bool = False
    patient_agent: Optional[PatientAgent] = None
    measurement_agent: Optional[MeasurementAgent] = None
    moderator_llm: Optional[str] = None
    diagnosis_action: Optional[DoctorAction] = None
    input_tensors: List[torch.LongTensor] = field(default_factory=list)
    response_tensors: List[torch.LongTensor] = field(default_factory=list)
    total_num_tokens: int = 0

    def add_turn(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

@dataclass
class DoctorAction:
    """Parsed representation of a doctor LLM action."""

    type: str
    text: str
    payload: Optional[str] = None
    normalized_payload: Optional[str] = None


# ---------------------------------------------------------------------------
# Scenario loading utilities
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    clean = text.lower()
    clean = re.sub(r"[^a-z0-9]+", " ", clean)
    return clean.strip()

# ---------------------------------------------------------------------------
# Interaction parsing and reward scaffolding
# ---------------------------------------------------------------------------


def parse_action(text: str) -> DoctorAction:
    clean_text = text.strip()
    diag_match = re.search(r"DIAGNOSIS\s*READY\s*:\s*(.+)", clean_text, flags=re.IGNORECASE)
    test_match = re.search(r"REQUEST\s+TEST\s*:\s*(.+)", clean_text, flags=re.IGNORECASE)

    if diag_match:
        diagnosis = diag_match.group(1).strip()
        return DoctorAction(
            type="diagnosis",
            text=clean_text,
            payload=diagnosis,
            normalized_payload=normalize_text(diagnosis),
        )
    elif test_match:
        test_name = test_match.group(1).strip()
        return DoctorAction(
            type="test_request",
            text=clean_text,
            payload=test_name,
            normalized_payload=normalize_text(test_name),
        )
    else:
        return DoctorAction(
            type="question",
            text=clean_text,
            payload=clean_text,
            normalized_payload=normalize_text(clean_text),
        )


class AgentClinicSimulator:
    """AgentClinic multi-agent simulator coordinating doctor, patient, measurement, moderator."""

    def __init__(
        self,
        scenario_loader: ScenarioLoaderMedQA,
        patient_backend: str,
        measurement_backend: str,
        moderator_backend: Optional[str],
        max_turns: int,
        question_cost: float,
        temporal_decay_beta: float,
        budget_reward_weight: float,
        question_reward_weight: float,
        diagnosis_reward_weight: float,
        seed: int = 0,
        debug_print: bool = False,
        reward_breakdown_debug: bool = False,
        reward_forward_sim: bool = False,
        intrinsic_token_weight: float = 0.0,
        intrinsic_turn_weight: float = 0.0,
        forward_sim_temperature: float = 0.0,
        reward_budget_aware: bool = False,
        target_num_turns: int = 3,
        turn_penalty_weight: float = 0.1,
        turn_reward_weight: float = 0.05,
        save_llm_outputs: bool = False,
        output_dir: Optional[str] = None,
        vllm_tensor_parallel_size: int = 1,
        vllm_gpu_memory_utilization: float = 0.9,
        vllm_verbose: bool = False,
    ) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")

        self.scenario_loader = scenario_loader
        self.patient_backend = patient_backend
        self.measurement_backend = measurement_backend
        self.moderator_backend = moderator_backend
        self.max_turns = max_turns
        self.question_cost = float(question_cost)
        self.temporal_decay_beta = float(temporal_decay_beta)
        self.budget_reward_weight = float(budget_reward_weight)
        self.question_reward_weight = float(question_reward_weight)
        self.diagnosis_reward_weight = float(diagnosis_reward_weight)
        self.random = random.Random(seed)
        self._utility_warning_emitted = False
        self.forbidden_retry_limit = 3
        self.debug_print = debug_print
        self.reward_breakdown_debug = reward_breakdown_debug
        self.reward_forward_sim = reward_forward_sim
        self.intrinsic_token_weight = float(intrinsic_token_weight)
        self.intrinsic_turn_weight = float(intrinsic_turn_weight)
        self.forward_sim_temperature = float(forward_sim_temperature)
        self.reward_budget_aware = reward_budget_aware
        self.target_num_turns = target_num_turns
        self.turn_penalty_weight = float(turn_penalty_weight)
        self.turn_reward_weight = float(turn_reward_weight)
        self.save_llm_outputs = save_llm_outputs
        self.output_dir = output_dir
        self.vllm_tensor_parallel_size = vllm_tensor_parallel_size
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self.vllm_verbose = vllm_verbose

    @staticmethod
    def _format_model_input(system_prompt: str, prompt: str) -> Tuple[str, str]:
        return system_prompt, prompt

    def _save_episode_output(
        self,
        state: DoctorEpisodeState,
        epoch: int,
        episode_num: int,
        episode_info: Dict[str, Any],
    ) -> None:
        """Save episode interactions to a text file for tracking changes over training."""
        if not self.save_llm_outputs or not self.output_dir:
            return

        # Create subdirectory for LLM outputs
        llm_outputs_dir = os.path.join(self.output_dir, "llm_outputs")
        os.makedirs(llm_outputs_dir, exist_ok=True)

        # Create filename with epoch, episode, and scenario info
        # Special epoch values: -1 = train eval, -2 = test eval
        if epoch == -1:
            epoch_str = "eval_train"
        elif epoch == -2:
            epoch_str = "eval_test"
        else:
            epoch_str = f"epoch_{epoch:03d}"

        filename = f"{epoch_str}_episode_{episode_num:04d}_scenario_{state.scenario_id:03d}.txt"
        filepath = os.path.join(llm_outputs_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            # Header with metadata
            f.write("=" * 80 + "\n")
            f.write("EPISODE OUTPUT LOG\n")
            f.write("=" * 80 + "\n")
            if epoch == -1:
                f.write("Epoch:              TRAIN EVALUATION\n")
            elif epoch == -2:
                f.write("Epoch:              TEST EVALUATION\n")
            else:
                f.write(f"Epoch:              {epoch}\n")
            f.write(f"Episode Number:     {episode_num}\n")
            f.write(f"Scenario ID:        {state.scenario_id}\n")
            f.write(f"Number of Turns:    {len(state.actions)}\n")
            f.write(f"Correctness:        {episode_info.get('correctness', 0.0):.3f}\n")
            reward = episode_info.get('reward', 0.0)
            total_reward = sum(reward) if isinstance(reward, list) else reward
            f.write(f"Total Reward:       {total_reward:.3f}\n")
            f.write(f"Budget Saved:       {episode_info.get('budget_saved', 0.0):.3f}\n")
            f.write("=" * 80 + "\n\n")

            # Scenario information
            f.write("SCENARIO INFORMATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Correct Diagnosis:  {state.scenario.diagnosis_information()}\n")
            f.write(f"Examiner Info:      {state.scenario.examiner_information()}\n")
            f.write("-" * 80 + "\n\n")

            # Turn-by-turn interactions
            f.write("INTERACTION HISTORY\n")
            f.write("=" * 80 + "\n\n")

            turn_num = 0
            for i, turn in enumerate(state.history):
                if turn["role"] == "doctor":
                    turn_num += 1
                    action = state.actions[turn_num - 1] if turn_num <= len(state.actions) else None
                    f.write("=" * 80 + "\n")
                    f.write(f"TURN {turn_num}\n")
                    f.write("=" * 80 + "\n")
                    if action:
                        f.write(f"Action Type: {action.type.upper()}\n")
                    f.write(f"\n[DOCTOR]\n{turn['content']}\n\n")
                else:
                    f.write(f"[{turn['role'].upper()}]\n{turn['content']}\n\n")

            # Final diagnosis (if provided)
            if state.diagnosis_action:
                f.write("=" * 80 + "\n")
                f.write("FINAL DIAGNOSIS\n")
                f.write("=" * 80 + "\n")
                f.write(f"{state.diagnosis_action.text}\n\n")
                f.write(f"Diagnosis Payload:  {state.diagnosis_action.payload}\n")
                f.write(f"Correct Diagnosis:  {state.scenario.diagnosis_information()}\n")
                f.write(f"Match:              {'YES' if episode_info.get('correctness', 0.0) > 0 else 'NO'}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF EPISODE\n")
            f.write("=" * 80 + "\n")

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self, scenario: ScenarioMedQA) -> DoctorEpisodeState:
        patient_agent = PatientAgent(
            scenario=scenario,
            backend_str=self.patient_backend,
            vllm_tensor_parallel_size=self.vllm_tensor_parallel_size,
            vllm_gpu_memory_utilization=self.vllm_gpu_memory_utilization,
            vllm_verbose=self.vllm_verbose,
        )
        measurement_agent = MeasurementAgent(
            scenario=scenario,
            backend_str=self.measurement_backend,
            vllm_tensor_parallel_size=self.vllm_tensor_parallel_size,
            vllm_gpu_memory_utilization=self.vllm_gpu_memory_utilization,
            vllm_verbose=self.vllm_verbose,
        )
        patient_agent.reset()
        measurement_agent.reset()

        return DoctorEpisodeState(
            scenario_id=scenario.scenario_id,
            scenario=scenario,
            remaining_budget=float(self.max_turns),
            max_turns=self.max_turns,
            history=[],
            actions=[],
            done=False,
            patient_agent=patient_agent,
            measurement_agent=measurement_agent,
            moderator_llm=self.moderator_backend,
            diagnosis_action=None,
        )

    def reset_by_id(self, scenario_id: int) -> DoctorEpisodeState:
        if not 0 <= scenario_id < self.scenario_loader.num_scenarios:
            raise IndexError(f"Scenario id {scenario_id} out of range")
        scenario = self.scenario_loader.get_scenario(id=scenario_id)
        return self.reset(scenario)

    def build_prompt_for_doctor(
        self,
        state: DoctorEpisodeState,
        forbidden_questions: Optional[Sequence[str]] = None,
    ) -> str:
        turns_taken = len(state.actions)
        turns_remaining = max(state.max_turns - turns_taken, 0)

        system_prompt = [
            "You are a doctor named Dr. Agent who only responds in the form of dialogue. You are inspecting a patient who you will ask questions in order to understand their disease.",
            f"You are only allowed to ask {state.max_turns} questions total before you must make a decision. You have asked {turns_taken} questions so far.",
            "You can request test results using the format \"REQUEST TEST: [test]\". For example, \"REQUEST TEST: Chest_X-Ray\".",
            "Your dialogue will only be 1-3 sentences in length.",
            "Once you have decided to make a diagnosis, first provide your complete reasoning process leading to the diagnosis.",
            "Explain your reasoning clearly and concisely. Then, on a new line, output the final result in the exact format: \"DIAGNOSIS READY: [diagnosis here]\".",
            "In each turn, you can do only one of the following: ask a question, request a test, or make a diagnosis."
        ]
        if forbidden_questions:
            formatted = ", ".join(f'"{q}"' for q in forbidden_questions)
            system_prompt.append(
                f"Do not ask the following question(s) during this interaction: {formatted}."
            )
        system_prompt.append("\n\nBelow is all of the information you have. {}\n\nRemember, you must discover their disease by asking them questions. You are also able to provide exams.".format(state.scenario.examiner_information()))
        system_prompt_str = " ".join(system_prompt)

        prompt = ["\nHere is a history of your dialogue: "]
        if state.history:
            history_str = [
                f"{turn['role'].capitalize()}: {turn['content']}" for turn in state.history
            ]
            prompt.append("; ".join(history_str))
        else:
            prompt.append("No dialogue yet.")

        if turns_remaining == 1:
            prompt.append("This is the final interaction. Do not ask further questions or request additional tests. Provide your complete reasoning process leading to the diagnosis. Then, on a new line, you must output the final result in the exact format: \"DIAGNOSIS READY: [diagnosis here]\".")
        prompt.append("\n\nNow, please continue your dialogue.\nDoctor: ")

        prompt_str = " ".join(prompt)

        return prompt_str, system_prompt_str

    def run_episode(
        self,
        scenario_id: int,
        generate_fn,
        epoch: int = 0,
        episode_num: int = 0,
    ) -> Tuple[DoctorEpisodeState, Dict[str, float]]:
        state = self.reset_by_id(scenario_id)

        states = []
        while not state.done:
            prompt, system_prompt = self.build_prompt_for_doctor(state)
            model_input = self._format_model_input(system_prompt, prompt)
            input_tensor, response_tensor, doctor_response = generate_fn(model_input)
            action = parse_action(doctor_response)
            state.actions.append(action)
            state.add_turn("doctor", action.text)
            state.input_tensors.append(input_tensor)
            state.response_tensors.append(response_tensor)
            state.total_num_tokens += len(response_tensor)

            states.append(copy.deepcopy(state))

            reply_role, reply_text = self._apply_action(state, action)

            if self.debug_print:
                turn_num = len(state.actions)
                print(
                    f"\n========== EPISODE {state.scenario_id} :: TURN {turn_num} ==========",
                    flush=True,
                )
                print("[System Prompt]\n" + system_prompt, flush=True)
                print("[Prompt]\n" + prompt, flush=True)
                print(
                    f"[Doctor -> Patient/Test]\n{doctor_response}",
                    flush=True,
                )
                if reply_role and reply_text:
                    print(
                        f"[{reply_role.capitalize()} -> Doctor]\n{reply_text}",
                        flush=True,
                    )
                print("=" * 60, flush=True)

        rewards, components = self._compute_episode_reward(states, generate_fn)
        episode_info = {"reward": rewards, **components}

        # Save episode outputs if enabled
        self._save_episode_output(state, epoch, episode_num, episode_info)

        return state.input_tensors, state.response_tensors, episode_info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_action(self, state: DoctorEpisodeState, action: DoctorAction) -> Tuple[Optional[str], str]:
        if action.type == "diagnosis":
            state.diagnosis_action = action
            state.done = True
            return None, ""
        elif action.type == "test_request":
            reply_role = "measurement"
            reply_text = state.measurement_agent.inference_measurement(action.text)
            state.patient_agent.add_hist(reply_text)
        else:
            reply_role = "patient"
            reply_text = state.patient_agent.inference_patient(action.text)
            state.measurement_agent.add_hist(reply_text)

        state.add_turn(reply_role, reply_text)
        state.remaining_budget = max(0.0, state.remaining_budget - self.question_cost)

        if len(state.actions) >= state.max_turns:
            state.done = True

        return reply_role, reply_text

    def _compute_episode_reward(
        self,
        states: List[DoctorEpisodeState],
        generate_fn,
    ) -> Tuple[float, Dict[str, Any]]:

        last_state = copy.deepcopy(states[-1])
        _, _ = self._apply_action(last_state, last_state.actions[-1])
        correctness = self._evaluate_correctness(last_state)[0]
        budget_saved = last_state.remaining_budget / float(max(last_state.max_turns, 1))

        # Forward simulation reward: per-turn extrinsic + intrinsic rewards
        if self.reward_forward_sim:
            if self.debug_print:
                print(
                    f"\n[Episode {states[0].scenario_id}] Computing forward simulation rewards "
                    f"for {len(states)} turns | actual_correctness={correctness:.3f}",
                    flush=True
                )

            per_turn_rewards = []
            forward_sim_correctness = []

            for turn_idx, state in enumerate(states):
                # Extrinsic reward: forward simulate from this turn
                extrinsic_correctness, total_num_tokens = self._forward_simulate_from_turn(
                    state, generate_fn
                )
                forward_sim_correctness.append(extrinsic_correctness)

                # Intrinsic reward: token count + turn penalty
                intrinsic_reward = (
                    -total_num_tokens * self.intrinsic_token_weight
                    # - self.intrinsic_turn_weight
                )

                # Combined reward for this turn
                turn_reward = (
                    self.diagnosis_reward_weight * extrinsic_correctness
                    + intrinsic_reward
                )
                per_turn_rewards.append(turn_reward)

                if self.debug_print:
                    print(
                        f"  [Turn {turn_idx+1}/{len(states)}] "
                        f"forward_sim_correct={extrinsic_correctness:.3f} | "
                        f"tokens={total_num_tokens} | "
                        f"intrinsic={intrinsic_reward:.4f} | "
                        f"extrinsic={self.diagnosis_reward_weight * extrinsic_correctness:.4f} | "
                        f"turn_reward={turn_reward:.4f}",
                        flush=True
                    )

            if self.debug_print or self.reward_breakdown_debug:
                print(
                    f"\n[Episode {state.scenario_id}] Forward sim SUMMARY | "
                    f"actual_correctness={correctness:.3f} | "
                    f"avg_forward_sim_correctness={sum(forward_sim_correctness)/len(forward_sim_correctness):.3f} | "
                    f"intrinsic_token_weight={self.intrinsic_token_weight:.4f} | "
                    f"intrinsic_turn_weight={self.intrinsic_turn_weight:.4f} | "
                    f"total_reward={sum(per_turn_rewards):.3f} | "
                    f"reward_per_turn={per_turn_rewards}"
                )
                print("=" * 80, flush=True)

            return per_turn_rewards, {
                "correctness": correctness,
                "budget_saved": budget_saved,
                "reward_per_turn": per_turn_rewards,
                "forward_sim_correctness_avg": sum(forward_sim_correctness) / len(forward_sim_correctness) if forward_sim_correctness else 0.0,
            }

        # Budget-aware reward: turn efficiency + confidence-based question utility
        if self.reward_budget_aware:
            if self.debug_print:
                print(
                    f"\n[Episode {states[0].scenario_id}] Computing budget-aware rewards "
                    f"for {len(states)} turns | actual_correctness={correctness:.3f} | "
                    f"target_turns={self.target_num_turns}",
                    flush=True
                )

            per_turn_rewards = []
            question_utilities = []

            # Compute confidence before first turn (with no history)
            conf_before = self._get_diagnosis_confidence(states[0], generate_fn)

            for turn_idx, state in enumerate(states[1:]):
                # Get confidence after this turn (with response included, before next question)
                conf_after = self._get_diagnosis_confidence(state, generate_fn)

                utility = conf_after - conf_before
                question_utilities.append(utility)

                # Apply temporal weighting: ((N-i)/N)^beta
                N = len(states)
                temporal_weight = ((N - turn_idx) / N) ** self.temporal_decay_beta

                # Per-turn reward is temporally weighted question utility
                turn_reward = temporal_weight * utility
                per_turn_rewards.append(turn_reward)

                if self.debug_print:
                    print(
                        f"  [Turn {turn_idx+1}/{len(states)}] "
                        f"conf_before={conf_before:.3f} | "
                        f"conf_after={conf_after:.3f} | "
                        f"utility={utility:.3f} | "
                        f"temporal_weight={temporal_weight:.3f} | "
                        f"turn_reward={turn_reward:.4f}",
                        flush=True
                    )

                conf_before = conf_after

            # Add turn efficiency reward/penalty to the final turn
            num_turns = len(states)
            turn_diff = num_turns - self.target_num_turns

            if turn_diff > 0:
                # Penalty for exceeding target
                turn_efficiency = -turn_diff * self.turn_penalty_weight
            else:
                # Reward for staying below target
                turn_efficiency = -turn_diff * self.turn_reward_weight

            # Add final correctness reward to the last turn
            final_turn_reward = (
                self.diagnosis_reward_weight * correctness +
                turn_efficiency
            )
            per_turn_rewards.append(final_turn_reward)

            if self.debug_print or self.reward_breakdown_debug:
                avg_utility = sum(question_utilities) / len(question_utilities) if question_utilities else 0.0
                print(
                    f"\n[Episode {states[0].scenario_id}] Budget-aware SUMMARY | "
                    f"actual_correctness={correctness:.3f} | "
                    f"num_turns={num_turns} vs target={self.target_num_turns} | "
                    f"turn_efficiency={turn_efficiency:.3f} | "
                    f"avg_question_utility={avg_utility:.3f} | "
                    f"total_reward={sum(per_turn_rewards):.3f} | "
                    f"reward_per_turn={per_turn_rewards}"
                )
                print("=" * 80, flush=True)

            return per_turn_rewards, {
                "correctness": correctness,
                "budget_saved": budget_saved,
                "reward_per_turn": per_turn_rewards,
                "avg_question_utility": sum(question_utilities) / len(question_utilities) if question_utilities else 0.0,
                "turn_efficiency": turn_efficiency,
                "num_turns": num_turns,
            }

        # Simple correctness-only reward (for baseline evaluation without forward simulation)
        # This just returns the final correctness without expensive forward simulation
        # Useful for evaluating models without training
        simple_reward = correctness * self.diagnosis_reward_weight

        if self.debug_print or self.reward_breakdown_debug:
            print(
                f"[Episode {states[-1].scenario_id}] Simple correctness reward | "
                f"correctness={correctness:.3f} | "
                f"reward={simple_reward:.3f}"
            )

        return simple_reward, {
            "correctness": correctness,
            "budget_saved": budget_saved,
        }

    def _forward_simulate_from_turn(
        self,
        state: DoctorEpisodeState,
        generate_fn,
    ) -> float:
        """
        Forward simulate episode from states_idx to get expected correctness.

        This implements per-turn credit assignment by simulating what happens
        if we start from this turn and continue to the end.

        Process:
        1. Create fresh state for same scenario
        2. Restore state from snapshot AFTER turn_idx (O(1) instead of O(turn_idx) replay!)
        3. Generate fresh turns [turn_idx+1, ...] until diagnosis (deterministic)
        4. Return final correctness

        Args:
            original_state: Completed episode state with all turns
            turn_idx: Turn index to simulate from (0-indexed)
            generate_fn: Generation function that takes (prompt, scenario_id, turn_idx)

        Returns:
            Correctness (0.0 or 1.0) of forward-simulated episode
        """

        # state ends with doctor's action. Apply action.
        starting_turn = len(state.actions)

        if self.debug_print:
            print(
                f"    [ForwardSim] Starting from turn {starting_turn}, scenario {state.scenario_id} | "
                f"temperature={self.forward_sim_temperature}",
                flush=True
            )

        action = state.actions[-1]
        _, _ = self._apply_action(state, action)

        sim_turns = 0
        while not state.done:
            # Build prompt (returns tuple: prompt, system_prompt)
            prompt_str, system_prompt_str = self.build_prompt_for_doctor(state)
            model_input = self._format_model_input(system_prompt_str, prompt_str)

            # Generate with forward_sim_temperature (typically 0.0 for deterministic)
            _, response_tensor, doctor_response = generate_fn(model_input, self.forward_sim_temperature)
            action = parse_action(doctor_response)
            state.actions.append(action)
            state.add_turn("doctor", action.text)
            state.total_num_tokens += len(response_tensor)

            sim_turns += 1
            if self.debug_print:
                print(f"    [ForwardSim] Sim turn {sim_turns}: {action.type} | tokens={len(response_tensor)}", flush=True)

            _, _ = self._apply_action(state, action)

        # Evaluate final correctness
        correctness, _ = self._evaluate_correctness(state)

        if self.debug_print:
            print(
                f"    [ForwardSim] Complete: simulated {sim_turns} turns | "
                f"total_tokens={state.total_num_tokens} | "
                f"correctness={correctness:.3f}",
                flush=True
            )

        return correctness, state.total_num_tokens

    def _evaluate_correctness(
        self, state: DoctorEpisodeState
    ) -> Tuple[float, str]:
        correctness = 0.0
        moderator_decision = ""
        if state.diagnosis_action:
            gold_text = state.scenario.diagnosis_information()
            if self.moderator_backend:
                moderator_decision = compare_results(
                    diagnosis=state.diagnosis_action.text,
                    correct_diagnosis=gold_text,
                    moderator_llm=self.moderator_backend,
                    vllm_tensor_parallel_size=self.vllm_tensor_parallel_size,
                    vllm_gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    vllm_verbose=self.vllm_verbose,
                )
                correctness = 1.0 if moderator_decision.strip().startswith("yes") else -0.5
        return correctness, moderator_decision

    def _get_diagnosis_confidence(
        self,
        state: DoctorEpisodeState,
        generate_fn,
    ) -> float:
        """
        Query the model for its confidence in the top diagnosis.

        Always excludes the most recent doctor question from history to measure
        confidence after receiving information but before asking the next question.

        Returns confidence score (0.0 = not confident, 1.0 = very confident).
        """
        # Build system prompt for confidence estimation (adapted from build_prompt_for_doctor)
        system_prompt = [
            "You are a doctor named Dr. Agent analyzing a patient interaction.",
            "Based on the information gathered, you need to assess your confidence in a diagnosis."
        ]
        system_prompt.append("\n\nPatient information:\n{}".format(state.scenario.examiner_information()))
        system_prompt_str = " ".join(system_prompt)

        # Build conversation history, ALWAYS excluding last doctor turn
        history_to_use = state.history[:]
        if history_to_use and history_to_use[-1]['role'] == 'doctor':
            history_to_use = history_to_use[:-1]

        # Create confidence assessment prompt
        confidence_prompt = ["\nConversation history so far:"]
        if history_to_use:
            history_str = [
                f"{turn['role'].capitalize()}: {turn['content']}" for turn in history_to_use
            ]
            confidence_prompt.append("\n".join(history_str))
        else:
            confidence_prompt.append("No dialogue yet.")

        confidence_prompt.append(
            "\n\nBased on the conversation above, assess your current understanding:\n"
            "1. Reasoning: Provide your analysis of what you know and what remains uncertain.\n"
            "2. Diagnosis: State your most likely diagnosis based on current information.\n"
            "3. Confidence: Rate your confidence in this diagnosis from 0.0 (not confident) to 1.0 (completely certain).\n\n"
            "Format:\n"
            "Reasoning: [your analysis]\n"
            "Diagnosis: [your diagnosis]\n"
            "Confidence: [0.0-1.0]\n\n"
            "Response:"
        )

        confidence_prompt_str = "\n".join(confidence_prompt)
        model_input = self._format_model_input(system_prompt_str, confidence_prompt_str)

        try:
            _, _, response = generate_fn(model_input, 0.0)  # Use temperature 0 for deterministic

            # Parse confidence score
            import re
            confidence_pattern = r'[Cc]onfidence\s*:?\s*([0-9]*\.?[0-9]+)'
            match = re.search(confidence_pattern, response)

            if match:
                confidence = float(match.group(1))
                # Clamp to [0, 1]
                confidence = max(0.0, min(1.0, confidence))

                if self.debug_print:
                    # Extract diagnosis for debugging
                    diag_pattern = r'[Dd]iagnosis\s*:?\s*([^\n]+)'
                    diag_match = re.search(diag_pattern, response)
                    diagnosis = diag_match.group(1).strip() if diag_match else "N/A"
                    print(f"    [Confidence] Diagnosis: {diagnosis} | Confidence: {confidence:.3f}", flush=True)

                return confidence
            else:
                # If parsing fails, return 0 confidence
                if self.debug_print:
                    print("    [Confidence] Failed to parse confidence from response", flush=True)
                return 0.0

        except Exception as e:
            if self.debug_print:
                print(f"    [Confidence] Error estimating confidence: {e}", flush=True)
            return 0.0  # Return 0 confidence on error


# ---------------------------------------------------------------------------
# Model loading utilities
# ---------------------------------------------------------------------------


def create_bnb_config(args) -> Optional[BitsAndBytesConfig]:
    if not args.use_4bit:
        return None
    if BitsAndBytesConfig is None:
        raise ImportError(
            "BitsAndBytesConfig is required for --use_4bit but the current transformers installation does not expose it."
        )
    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )


def log_trainable_parameters(model: AutoModelForCausalLMWithValueHead) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    percentage = 100.0 * trainable / total if total else 0.0
    logger.info(
        "Trainable parameters: %s/%s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        percentage,
    )


def load_model_and_tokenizer(
    args,
    bnb_config: Optional[BitsAndBytesConfig],
    torch_dtype: Optional[torch.dtype],
) -> Tuple[AutoModelForCausalLMWithValueHead, AutoTokenizer]:
    load_source = args.resume_ckpt_dir or args.model_name or args.base_model_name
    device_map = "auto" if args.device == "auto" else {"": args.device}

    load_source = load_source.replace("HF_", "")
    logger.info("Loading policy model from %s", load_source)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        load_source,
        device_map=device_map,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )

    if args.use_lora:
        target_modules = [module.strip() for module in args.target_modules.split(",") if module.strip()]
        lora_config = LoraConfig(
            r=args.peft_r,
            lora_alpha=args.peft_alpha,
            lora_dropout=args.peft_dropout,
            bias="none",
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )
        model.pretrained_model = get_peft_model(model.pretrained_model, lora_config)
        logger.info(
            "Enabled LoRA adapters (r=%s, alpha=%s, dropout=%s) over modules: %s",
            args.peft_r,
            args.peft_alpha,
            args.peft_dropout,
            ", ".join(target_modules) or "<auto>",
        )

    tokenizer_name = args.tokenizer_name or args.base_model_name
    tokenizer_name = tokenizer_name.replace("HF_", "")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=not args.disable_fast_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )

    added_tokens = 0
    if tokenizer.pad_token is None:
        pad_token = tokenizer.eos_token or "<|pad|>"
        tokenizer.add_special_tokens({"pad_token": pad_token})
        added_tokens = 1
    tokenizer.padding_side = "left"

    if added_tokens > 0:
        model.pretrained_model.resize_token_embeddings(len(tokenizer))

    model.pretrained_model.config.use_cache = False
    if args.gradient_checkpointing:
        model.pretrained_model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing on the policy model.")

    log_trainable_parameters(model)
    return model, tokenizer


def load_reference_model(
    args,
    bnb_config: Optional[BitsAndBytesConfig],
    torch_dtype: Optional[torch.dtype],
) -> Optional[AutoModelForCausalLMWithValueHead]:
    if args.disable_reference_model:
        logger.info("Reference model disabled via --disable_reference_model")
        return None

    ref_source = args.ref_model_name or args.base_model_name
    device_map = "auto" if args.device == "auto" else {"": args.device}
    ref_source = ref_source.replace("HF_", "")
    logger.info("Loading reference model from %s", ref_source)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        ref_source,
        device_map=device_map,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    ref_model.pretrained_model.config.use_cache = False
    ref_model.eval()
    ref_model.requires_grad_(False)
    return ref_model


# ---------------------------------------------------------------------------
# Checkpoint persistence
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def prepare_generation_kwargs(args, tokenizer: AutoTokenizer) -> Dict:
    return {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": args.temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }


def train(args) -> None:
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    set_seed(args.seed)

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

    if not train_indices:
        raise ValueError("No training scenarios available after train/test split.")

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
        reward_breakdown_debug=args.print_reward_breakdown or args.debug_print,
        reward_forward_sim=args.reward_forward_sim,
        intrinsic_token_weight=args.intrinsic_token_weight,
        intrinsic_turn_weight=args.intrinsic_turn_weight,
        forward_sim_temperature=args.forward_sim_temperature,
        reward_budget_aware=args.reward_budget_aware,
        target_num_turns=args.target_num_turns,
        turn_penalty_weight=args.turn_penalty_weight,
        turn_reward_weight=args.turn_reward_weight,
        save_llm_outputs=args.save_llm_outputs,
        output_dir=output_dir,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_verbose=args.vllm_verbose,
    )

    bnb_config = create_bnb_config(args)
    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)

    model, tokenizer = load_model_and_tokenizer(args, bnb_config, torch_dtype)
    ref_model = load_reference_model(args, bnb_config, torch_dtype)

    ppo_config = PPOConfig(
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        ppo_epochs=args.num_ppo_epochs,
        remove_unused_columns=False,
        is_peft_model=args.use_lora,
    )

    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=None,
        data_collator=None,
    )

    generation_kwargs = prepare_generation_kwargs(args, tokenizer)
    checkpoint_manager = CheckpointManager(output_dir, args.save_total_limit)

    device = trainer.accelerator.device
    policy_model = trainer.accelerator.unwrap_model(trainer.model).pretrained_model

    # Optionally load vLLM for faster policy generation during training
    policy_vllm = None
    if args.use_vllm_policy:
        if LLM is None:
            raise ImportError("vLLM is not installed. Install it with: pip install vllm")
        # Suppress vLLM logging unless verbose mode is enabled
        if not args.vllm_verbose:
            suppress_vllm_logging()
        logger.info("Loading vLLM for policy model generation (training)")
        model_source = args.model_name or args.base_model_name
        model_source = model_source.replace("HF_", "")
        policy_vllm = LLM(
            model=model_source,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            trust_remote_code=args.trust_remote_code,
            disable_log_stats=not args.vllm_verbose,  # Disable progress bars unless verbose mode
        )
        logger.info("vLLM policy model loaded successfully")

    def generate_response(
        prompt: Tuple[str, str],
        temperature_override: Optional[float] = None,
    ) -> Tuple[torch.LongTensor, torch.LongTensor, str]:
        system_prompt_text, user_prompt_text = prompt

        chat_messages: List[Dict[str, str]] = []
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

        # Override temperature for forward simulation if specified
        temperature = temperature_override if temperature_override is not None else args.temperature

        if args.debug_print and temperature_override is None:
            # Only print once per episode to avoid spam
            if not hasattr(generate_response, '_temp_printed'):
                print(f"[Generation] Using temperature={temperature}, use_vllm={policy_vllm is not None}", flush=True)
                generate_response._temp_printed = True

        if policy_vllm is not None:
            # Use vLLM for generation (faster)
            if SamplingParams is None:
                raise ImportError("vLLM is not installed. Install it with: pip install vllm")

            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=args.max_new_tokens,
                top_p=args.top_p,
            )

            # Generate with vLLM and get token IDs directly
            outputs = policy_vllm.generate([prompt_for_model], sampling_params)
            output = outputs[0]

            # Extract prompt and completion token IDs
            prompt_token_ids = output.prompt_token_ids
            completion_token_ids = output.outputs[0].token_ids

            # Convert to tensors
            input_tensor = torch.LongTensor(prompt_token_ids)
            response_tensor = torch.LongTensor(completion_token_ids)

            # Decode response text
            response_text = tokenizer.decode(completion_token_ids, skip_special_tokens=True).strip()

            return input_tensor, response_tensor, response_text
        else:
            # Use HuggingFace model for generation
            inputs = tokenizer(prompt_for_model, return_tensors="pt").to(device)
            query_tensors = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")

            gen_kwargs = generation_kwargs.copy()
            if temperature_override is not None:
                gen_kwargs["temperature"] = temperature_override
                gen_kwargs["do_sample"] = temperature_override > 0

            # Keep model in training mode during PPO rollout (following CollabLLM)
            # This ensures policy consistency between data collection and optimization
            with torch.no_grad():
                output_tensors = policy_model.generate(
                    query_tensors,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )

            generated_tokens = output_tensors[:, query_tensors.shape[-1]:]
            if generated_tokens.shape[-1] == 0:
                generated_tokens = output_tensors[:, -1:]

            input_tensor = query_tensors.squeeze(0).detach()
            response_tensor = generated_tokens.squeeze(0).detach()
            response_text = tokenizer.decode(response_tensor, skip_special_tokens=True).strip()

            return input_tensor, response_tensor, response_text

    global_step = 0
    episodes_completed = 0
    stop_training = False
    episode_metrics: List[Dict[str, float]] = []
    epoch_stats: List[Dict[str, Any]] = []  # Track stats per epoch

    # Ensure model is in training mode for PPO (following CollabLLM)
    policy_model.train()

    for epoch in tqdm(range(args.num_train_epochs), desc="Epochs", position=0, disable=not trainer.accelerator.is_main_process):
        scenario_indices = train_indices.copy()
        random.shuffle(scenario_indices)
        logger.info("Starting epoch %s with %s training scenarios (scenarios_per_update=%s)", epoch + 1, len(scenario_indices), args.scenarios_per_update)

        epoch_rewards: List[float] = []
        epoch_correctness: List[float] = []
        epoch_turns: List[int] = []

        scenario_pbar = tqdm(
            scenario_indices,
            desc=f"Epoch {epoch+1}/{args.num_train_epochs} - Scenarios",
            position=1,
            leave=False,
            disable=not trainer.accelerator.is_main_process
        )

        # Batch accumulators for multiple scenarios
        batch_input_tensors: List[torch.LongTensor] = []
        batch_response_tensors: List[torch.LongTensor] = []
        batch_reward_tensors: List[torch.Tensor] = []
        batch_episode_infos: List[Dict[str, Any]] = []
        batch_scenario_ids: List[int] = []

        for scenario_idx in scenario_pbar:
            input_tensors, response_tensors, episode_info = simulator.run_episode(
                scenario_idx, generate_response, epoch=epoch + 1, episode_num=episodes_completed
            )

            multi_turn_rewards = episode_info.pop("reward")
            reward_value = sum(multi_turn_rewards)
            reward_tensors = [torch.tensor(r, device=device, dtype=torch.float32) for r in multi_turn_rewards]

            # Accumulate data from this scenario
            batch_input_tensors.extend(input_tensors)
            batch_response_tensors.extend(response_tensors)
            batch_reward_tensors.extend(reward_tensors)
            batch_episode_infos.append({**episode_info, "reward": reward_value, "scenario_id": scenario_idx})
            batch_scenario_ids.append(scenario_idx)

            # Track individual episode metrics
            episode_metrics.append(
                {
                    "correctness": episode_info.get("correctness", 0.0),
                    "num_turns": len(input_tensors),
                }
            )
            epoch_rewards.append(reward_value)
            epoch_correctness.append(episode_info.get("correctness", 0.0))
            epoch_turns.append(len(input_tensors))

            episodes_completed += 1

            # Check if we've accumulated enough scenarios for a batch update
            if len(batch_scenario_ids) >= args.scenarios_per_update or scenario_idx == scenario_indices[-1]:
                # Update batch_size to match the total number of turns across all accumulated scenarios
                trainer.config.batch_size = len(batch_input_tensors)

                # Single trainer.step() call with all turns from all accumulated scenarios
                stats = trainer.step(batch_input_tensors, batch_response_tensors, batch_reward_tensors)

                # Prepare batch data for logging (use first scenario as representative)
                batch_data = {
                    "scenario_id": batch_scenario_ids[0],
                    "input_tensors": batch_input_tensors,
                    "response_tensors": batch_response_tensors,
                    "reward_tensors": batch_reward_tensors,
                }

                # Log stats once for the entire batch
                trainer.log_stats(stats, batch_data, batch_reward_tensors)

                # Gradient verification once per batch
                if args.debug_verify_gradients and trainer.accelerator.is_main_process:
                    verify_ppo_gradients(
                        trainer,
                        stats,
                        context=f"epoch={epoch + 1},scenarios={batch_scenario_ids},num_turns={len(batch_input_tensors)}",
                    )

                # Log per-episode info for each scenario in the batch
                if trainer.accelerator.is_main_process:
                    for ep_info in batch_episode_infos:
                        scalar_logs = {
                            f"episode/{key}": value
                            for key, value in ep_info.items()
                            if isinstance(value, (int, float))
                        }
                        trainer.accelerator.log(scalar_logs)

                global_step += len(batch_input_tensors)

                # Clear batch accumulators
                batch_input_tensors = []
                batch_response_tensors = []
                batch_reward_tensors = []
                batch_episode_infos = []
                batch_scenario_ids = []

            # Update progress bar with running metrics
            if trainer.accelerator.is_main_process:
                avg_reward = sum(epoch_rewards) / len(epoch_rewards)
                avg_correctness = sum(epoch_correctness) / len(epoch_correctness)
                avg_turns = sum(epoch_turns) / len(epoch_turns)
                scenario_pbar.set_postfix({
                    'reward': f'{avg_reward:.3f}',
                    'correct': f'{avg_correctness:.2%}',
                    'turns': f'{avg_turns:.1f}'
                })

            if args.logging_steps and global_step % args.logging_steps == 0:
                logger.info(
                    "step=%s epoch=%s episodes_completed=%s avg_reward=%.3f avg_correctness=%.3f",
                    global_step,
                    epoch + 1,
                    episodes_completed,
                    sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0,
                    sum(epoch_correctness) / len(epoch_correctness) if epoch_correctness else 0.0,
                )

            if args.max_train_steps and global_step >= args.max_train_steps:
                stop_training = True

            if args.max_episodes and episodes_completed >= args.max_episodes:
                stop_training = True

            if args.save_steps and global_step % args.save_steps == 0:
                save_checkpoint(
                    trainer, tokenizer, output_dir, f"step_{global_step}", checkpoint_manager
                )

            if stop_training:
                break

        # Compute and log epoch-level statistics
        if epoch_rewards:
            epoch_avg_reward = sum(epoch_rewards) / len(epoch_rewards)
            epoch_avg_accuracy = sum(epoch_correctness) / len(epoch_correctness)
            epoch_avg_interactions = sum(epoch_turns) / len(epoch_turns)

            epoch_stat = {
                "epoch": epoch + 1,
                "avg_reward": epoch_avg_reward,
                "avg_accuracy": epoch_avg_accuracy,
                "avg_interactions": epoch_avg_interactions,
                "num_episodes": len(epoch_rewards),
            }
            epoch_stats.append(epoch_stat)

            logger.info(
                "Epoch %d summary: avg_reward=%.3f, avg_accuracy=%.3f, avg_interactions=%.2f",
                epoch + 1,
                epoch_avg_reward,
                epoch_avg_accuracy,
                epoch_avg_interactions,
            )

        if stop_training:
            break

    # Summarise training-set accuracy/interaction statistics
    if episode_metrics and trainer.accelerator.is_main_process:
        total_correct = sum(m["correctness"] for m in episode_metrics)
        accuracy = total_correct / len(episode_metrics)
        avg_turns = sum(m["num_turns"] for m in episode_metrics) / len(episode_metrics)
        logger.info(
            "Training episodes: accuracy=%.3f (%s/%s), avg_interactions=%.2f",
            accuracy,
            int(total_correct),
            len(episode_metrics),
            avg_turns,
        )

    # Evaluate final policy on train and test sets
    if trainer.accelerator.is_main_process:
        was_training = policy_model.training
        policy_model.eval()

        # Evaluate on training set
        train_eval_metrics: List[Dict[str, float]] = []
        logger.info("Evaluating on %d training scenarios...", len(train_indices))
        for eval_idx, scenario_idx in enumerate(tqdm(train_indices, desc="Train Evaluation", disable=not trainer.accelerator.is_main_process)):
            input_tensors, response_tensors, episode_info = simulator.run_episode(
                scenario_idx, generate_response, epoch=-1, episode_num=eval_idx
            )
            train_eval_metrics.append(
                {
                    "correctness": episode_info.get("correctness", 0.0),
                    "num_turns": len(input_tensors),
                }
            )

        if train_eval_metrics:
            total_correct = sum(m["correctness"] for m in train_eval_metrics)
            train_accuracy = total_correct / len(train_eval_metrics)
            train_avg_turns = sum(m["num_turns"] for m in train_eval_metrics) / len(train_eval_metrics)
            logger.info(
                "[TRAIN EVAL] accuracy=%.3f (%s/%s), avg_interactions=%.2f",
                train_accuracy,
                int(total_correct),
                len(train_eval_metrics),
                train_avg_turns,
            )

        # Evaluate on test set if available
        if test_indices:
            test_eval_metrics: List[Dict[str, float]] = []
            logger.info("Evaluating on %d test scenarios...", len(test_indices))
            for eval_idx, scenario_idx in enumerate(tqdm(test_indices, desc="Test Evaluation", disable=not trainer.accelerator.is_main_process)):
                input_tensors, response_tensors, episode_info = simulator.run_episode(
                    scenario_idx, generate_response, epoch=-2, episode_num=eval_idx
                )
                test_eval_metrics.append(
                    {
                        "correctness": episode_info.get("correctness", 0.0),
                        "num_turns": len(input_tensors),
                    }
                )

            if test_eval_metrics:
                total_correct = sum(m["correctness"] for m in test_eval_metrics)
                test_accuracy = total_correct / len(test_eval_metrics)
                test_avg_turns = sum(m["num_turns"] for m in test_eval_metrics) / len(test_eval_metrics)
                logger.info(
                    "[TEST EVAL] accuracy=%.3f (%s/%s), avg_interactions=%.2f",
                    test_accuracy,
                    int(total_correct),
                    len(test_eval_metrics),
                    test_avg_turns,
                )

        # Save final evaluation results in standard format for sweep compatibility
        eval_results_summary = {}
        if train_eval_metrics:
            total_correct = sum(m["correctness"] for m in train_eval_metrics)
            eval_results_summary["train"] = {
                "accuracy": total_correct / len(train_eval_metrics),
                "avg_interactions": sum(m["num_turns"] for m in train_eval_metrics) / len(train_eval_metrics),
                "num_scenarios": len(train_indices),
                "num_correct": int(total_correct),
            }
        if test_indices and test_eval_metrics:
            total_correct = sum(m["correctness"] for m in test_eval_metrics)
            eval_results_summary["test"] = {
                "accuracy": total_correct / len(test_eval_metrics),
                "avg_interactions": sum(m["num_turns"] for m in test_eval_metrics) / len(test_eval_metrics),
                "num_scenarios": len(test_indices),
                "num_correct": int(total_correct),
            }
        if eval_results_summary:
            eval_path = os.path.join(output_dir, "evaluation_results.json")
            with open(eval_path, "w", encoding="utf-8") as f:
                json.dump(eval_results_summary, f, indent=2)
            logger.info("Saved evaluation results to %s", eval_path)

        if was_training:
            policy_model.train()

    trainer.accelerator.wait_for_everyone()
    if getattr(trainer.accelerator, "is_main_process", True):
        trainer.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("Saved final policy and tokenizer to %s", output_dir)

        # Save epoch statistics to JSON file for later analysis/plotting
        if epoch_stats:
            stats_path = os.path.join(output_dir, "epoch_stats.json")
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(epoch_stats, f, indent=2)
            logger.info("Saved epoch statistics to %s", stats_path)


# ---------------------------------------------------------------------------
# CLI utilities
# ---------------------------------------------------------------------------


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("AgentClinic PPO fine-tuning")
    parser.add_argument("--dataset_path", type=str, default="agentclinic_medqa.jsonl", help="Path to AgentClinic JSONL dataset")
    parser.add_argument("--max_scenarios", type=int, default=None, help="Optional cap on scenarios for quicker iterations")
    parser.add_argument("--test_size", type=int, default=None, help="Number of scenarios to hold out for test set")
    parser.add_argument("--test_ratio", type=float, default=None, help="Ratio of scenarios to hold out for test set (e.g., 0.2 for 20%)")
    parser.add_argument("--output_dir", type=str, default="outputs/ppo_run", help="Directory to store checkpoints and final policy")
    parser.add_argument("--base_model_name", type=str, default="HF_Qwen/Qwen2.5-7B-Instruct", help="Base HF model id for initialisation")
    parser.add_argument("--model_name", type=str, default=None, help="Optional SFT checkpoint to initialise from")
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Tokenizer identifier (defaults to base model)")
    parser.add_argument("--ref_model_name", type=str, default=None, help="Reference model for KL penalty (defaults to base)")
    parser.add_argument("--disable_reference_model", action="store_true", help="Skip loading a reference model (disables KL term)")
    parser.add_argument("--resume_ckpt_dir", type=str, default=None, help="Resume policy weights from a previous PPO checkpoint")

    parser.add_argument("--patient_llm", type=str, default="HF_Qwen/Qwen2.5-7B-Instruct", help="LLM backend for the patient agent (prefixed with HF_ or VLLM_)")
    parser.add_argument("--measurement_llm", type=str, default="HF_Qwen/Qwen2.5-7B-Instruct", help="LLM backend for the measurement agent (prefixed with HF_ or VLLM_)")
    parser.add_argument("--moderator_llm", type=str, default="HF_Qwen/Qwen2.5-7B-Instruct", help="LLM backend for moderator rewards (prefixed with HF_ or VLLM_)")

    parser.add_argument("--use_vllm_policy", action="store_true", help="Use vLLM for policy model generation during training (significantly faster inference)")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1, help="Number of GPUs to use for vLLM tensor parallelism")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization for vLLM (0.0-1.0)")
    parser.add_argument("--vllm_verbose", action="store_true", help="Enable verbose logging for vLLM (default: suppressed)")

    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA adapters for efficient fine-tuning")
    parser.add_argument("--peft_r", type=int, default=32)
    parser.add_argument("--peft_alpha", type=int, default=16)
    parser.add_argument("--peft_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names for LoRA injection",
    )

    parser.add_argument("--use_4bit", action="store_true", help="Load model with 4-bit quantisation (bitsandbytes)")
    parser.add_argument("--bf16", action="store_true", help="Load weights in bfloat16 where supported")
    parser.add_argument("--fp16", action="store_true", help="Load weights in float16")
    parser.add_argument("--device", type=str, default="auto", help="Device map hint (e.g. 'cuda', 'cpu', 'auto')")
    parser.add_argument("--trust_remote_code", action="store_true", help="Allow custom code for model/tokenizer loading")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", help="Force use of slow tokenizer implementation")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing on the policy model")

    parser.add_argument("--scenarios_per_update", type=int, default=1, help="Number of scenarios to accumulate before each PPO update")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of episodes per PPO update (automatically set based on scenarios_per_update)")
    parser.add_argument("--mini_batch_size", type=int, default=1, help="PPO mini-batch size (currently forced to 1)")
    parser.add_argument("--num_ppo_epochs", type=int, default=4, help="Number of PPO optimisation epochs per batch")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of passes over the scenario list")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    parser.add_argument("--max_turns", type=int, default=5, help="Maximum doctor turns before forced diagnosis")
    parser.add_argument("--question_cost", type=float, default=1.0, help="Budget cost per non-diagnosis action")
    parser.add_argument("--temporal_decay_beta", type=float, default=1.0, help="Temporal decay exponent β for question rewards ( (N-i)/N )^β")
    parser.add_argument("--budget_reward_weight", type=float, default=1.0, help="Weight applied to budget fraction in episode reward")
    parser.add_argument("--question_reward_weight", type=float, default=1.0, help="Weight applied to the summed question utilities")
    parser.add_argument("--diagnosis_reward_weight", type=float, default=1.0, help="Weight applied to diagnosis correctness")
    parser.add_argument("--debug_print", action="store_true", help="Print interactions and counterfactual details for debugging")
    parser.add_argument("--print_reward_breakdown", action="store_true", help="Print per-turn reward component breakdowns during training")
    parser.add_argument("--debug_verify_gradients", action="store_true", help="Run PPO gradient health checks after each optimisation step")
    parser.add_argument("--save_llm_outputs", action="store_true", help="Save LLM interactions to output directory for tracking changes over training")
    parser.add_argument("--reward_forward_sim", action="store_true", help="Use forward simulation rewards: per-turn extrinsic (forward sim correctness) + intrinsic (token/turn penalties)")
    parser.add_argument("--intrinsic_token_weight", type=float, default=0.001, help="Penalty weight per token in intrinsic reward (e.g., 0.001)")
    parser.add_argument("--intrinsic_turn_weight", type=float, default=0.0, help="Flat penalty per turn in intrinsic reward")
    parser.add_argument("--forward_sim_temperature", type=float, default=0.0, help="Temperature for forward simulation (0.0 = deterministic)")

    parser.add_argument("--reward_budget_aware", action="store_true", help="Use budget-aware rewards: turn efficiency + confidence-based question utility")
    parser.add_argument("--target_num_turns", type=int, default=3, help="Target number of interactions for budget-aware reward")
    parser.add_argument("--turn_penalty_weight", type=float, default=0.1, help="Penalty weight per turn above target")
    parser.add_argument("--turn_reward_weight", type=float, default=0.05, help="Reward weight per turn below target")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=0, help="Checkpoint every N PPO updates (0 disables)")
    parser.add_argument("--save_total_limit", type=int, default=5, help="Maximum number of saved checkpoints (excluding final)")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Optional cap on PPO updates")
    parser.add_argument("--max_episodes", type=int, default=None, help="Optional cap on completed episodes")

    parser.add_argument("--config_file", type=str, default=None, help="JSON or YAML file with argument overrides")
    parser.add_argument("--log_level", type=str, default="INFO")

    args = parser.parse_args()

    if args.config_file:
        with open(args.config_file, "r", encoding="utf-8") as handle:
            if args.config_file.endswith((".yml", ".yaml")):
                import yaml  # Lazy import to avoid hard dependency when unused

                overrides = yaml.safe_load(handle)
            else:
                overrides = json.load(handle)
        for key, value in overrides.items():
            if hasattr(args, key):
                setattr(args, key, value)

    return args


def validate_args(args) -> None:
    if args.bf16 and args.fp16:
        raise ValueError("Only one of --bf16 or --fp16 can be specified.")
    if args.scenarios_per_update <= 0:
        raise ValueError("--scenarios_per_update must be positive")
    if args.mini_batch_size != 1:
        logger.warning(
            "Overriding mini_batch_size=%s → 1 (turns within episode are batched together)",
            args.mini_batch_size,
        )
        args.mini_batch_size = 1
    if args.max_turns <= 0:
        raise ValueError("--max_turns must be positive")
    if args.question_cost < 0:
        raise ValueError("--question_cost must be non-negative")
    if args.save_steps < 0:
        raise ValueError("--save_steps must be non-negative")
    if args.save_total_limit is not None and args.save_total_limit < 0:
        raise ValueError("--save_total_limit must be non-negative")
    if args.temporal_decay_beta < 0:
        raise ValueError("--temporal_decay_beta must be non-negative")
    if args.test_size is not None and args.test_ratio is not None:
        raise ValueError("Cannot specify both --test_size and --test_ratio; choose one.")
    if args.test_size is not None and args.test_size <= 0:
        raise ValueError("--test_size must be positive")
    if args.test_ratio is not None and (args.test_ratio <= 0 or args.test_ratio >= 1):
        raise ValueError("--test_ratio must be between 0 and 1")
    for weight_name in (
        "budget_reward_weight",
        "question_reward_weight",
        "diagnosis_reward_weight",
    ):
        if getattr(args, weight_name) < 0:
            raise ValueError(f"--{weight_name} must be non-negative")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    validate_args(args)
    train(args)


if __name__ == "__main__":
    main()

