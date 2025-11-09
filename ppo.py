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
import shutil
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

try:  # Optional dependency for 4-bit loading
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - optional dependency
    BitsAndBytesConfig = None  # type: ignore

from transformers import AutoTokenizer, set_seed, pipeline

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from peft import LoraConfig, get_peft_model

try:  # Optional logging backend
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None  # type: ignore


logger = logging.getLogger(__name__)


def _stringify_context(value) -> str:
    if isinstance(value, dict):
        return "; ".join(f"{k}: {_stringify_context(v)}" for k, v in value.items())
    if isinstance(value, list):
        return "; ".join(_stringify_context(v) for v in value)
    return str(value)


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

def query_model(model, prompt, system_prompt, tries=30, timeout=20.0, max_prompt_len=2**14, clip_prompt=False):
    for _ in range(tries):
        if clip_prompt: 
            prompt = prompt[:max_prompt_len]
        try:
            if isinstance(model, str):
                if not model.startswith("HF_"):
                    raise ValueError("Only HuggingFace backends prefixed with 'HF_' are currently supported.")
                # Extract the HF repo id
                hf_id = model[3:]
                
                # Load or retrieve cached pipeline
                pipe = HUGGINGFACE_PIPES.get(hf_id)
                if pipe is None:
                    pipe = load_huggingface_model(hf_id)
                    HUGGINGFACE_PIPES[hf_id] = pipe
            else:
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
                # print(f'input_text: {input_text}')
                # print('chat template applied!')
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
    def __init__(self, path: str = "agentclinic_medqa.jsonl", max_scenarios: Optional[int] = None) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            scenario_strs = [json.loads(line) for line in f if line.strip()]

        if max_scenarios is not None:
            scenario_strs = scenario_strs[:max_scenarios]

        self.scenarios = [ScenarioMedQA(data, idx) for idx, data in enumerate(scenario_strs)]
        self.num_scenarios = len(self.scenarios)

    def sample_scenario(self):
        return random.choice(self.scenarios)

    def get_scenario(self, id):
        if id is None:
            return self.sample_scenario()
        return self.scenarios[id]

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
class PatientAgent:
    def __init__(self, scenario, backend_str="gpt4", bias_present=None) -> None:
        # disease of patient, or "correct answer"
        self.disease = ""
        # symptoms that patient presents
        self.symptoms = ""
        # conversation history between doctor and patient
        self.agent_hist = ""
        # language model backend for patient agent
        self.backend = backend_str
        # presentation of any form of bias
        self.bias_present = (None if bias_present == "None" else bias_present)
        # sample initial question from dataset
        self.scenario = scenario
        self.reset()
        self.pipe = None

        self.biases = ["recency", "frequency", "false_consensus", "self_diagnosis", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"]

    def generate_bias(self) -> str:
        """ 
        ================
        Cognitive biases 
        ================
        """
        if self.bias_present == "recency":
            return "\nRecently, you had a friend with similar symptoms to yours that was diagnosed with cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "frequency":
            return "\nYou recently learned that most frequently, doctors diagnose this as cancer based on past cases. This affects how you interact with the doctor.\n"
        elif self.bias_present == "false_consensus":
            return "\nMost of your friends who heard about your symptoms believe that you have cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "self_diagnosis":
            return "\nYou recently performed a thorough investigation of your symptoms online and have come to the conclusion that you should be diagnosed with cancer. While the doctor may believe otherwise you remain convinced you have cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "gender":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular gender and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "race":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular race and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "sexual_orientation":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular sexual orientation and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "cultural":
            return "\nYou are uncomfortable with your doctor because you find out that they are from a particular culture and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "education":
            return "\nYou are uncomfortable with your doctor because you find out that they went to a low ranked medical school and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "religion":
            return "\nYou are uncomfortable with your doctor because you find out that they believe in a particular religion and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "socioeconomic":
            return "\nYou are uncomfortable with your doctor because you find out that they are from a particular socioeconomic background and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present is None:
            pass
        else:
            print("BIAS TYPE {} NOT SUPPORTED, ignoring bias...".format(self.bias_present))
        return ""

    def inference_patient(self, question) -> str:
        answer = query_model(self.backend, "\nHere is a history of your dialogue: " + self.agent_hist + "\n Here was the doctor response: " + question + "Now please continue your dialogue\nPatient: ", self.system_prompt())
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def system_prompt(self) -> str:
        bias_prompt = ""
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()
        base = """You are a patient in a clinic who only responds in the form of dialogue. You are being inspected by a doctor who will ask you questions and will perform exams on you in order to understand your disease. Your answer will only be 1-3 sentences in length."""
        symptoms = "\n\nBelow is all of your information. {}. \n\n Remember, you must not reveal your disease explicitly but may only convey the symptoms you have in the form of dialogue if you are asked.".format(self.symptoms)
        return base + bias_prompt + symptoms
    
    def reset(self) -> None:
        self.agent_hist = ""
        self.symptoms = self.scenario.patient_information()

    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"

class MeasurementAgent:
    def __init__(self, scenario, backend_str="gpt4") -> None:
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

    def inference_measurement(self, question) -> str:
        answer = str()
        answer = query_model(self.backend, "\nHere is a history of the dialogue: " + self.agent_hist + "\n Here was the doctor measurement request: " + question, self.system_prompt())
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


def compare_results(diagnosis, correct_diagnosis, moderator_llm):
    answer = query_model(moderator_llm, "\nHere is the correct diagnosis: " + correct_diagnosis + "\n Here was the doctor dialogue: " + diagnosis + "\nAre these the same?", "You are responsible for determining if the corrent diagnosis and the doctor diagnosis are the same disease. Please respond only with Yes or No. Nothing else.")
    return answer.lower()


def generate_doctor_bias_prompt(bias_type: Optional[str]) -> str:
    if not bias_type:
        return ""

    prompts = {
        "recency": "Recently, you learned that a doctor with similar symptoms diagnosed their patient with cancer. This affects how you interact with the patient.",
        "frequency": "You recently learned that most frequently, doctors diagnose patients with similar symptoms with cancer based on past cases. This affects how you interact with the patient.",
        "false_consensus": "Most of your colleagues who heard about the patient's symptoms believe that they have cancer. This affects how you interact with the patient.",
        "status_quo": "Your hospital has been diagnosing these patient's symptoms with cancer for the past 10 years. This affects how you interact with the patient.",
        "confirmation": "You are initially confident that the patient has cancer. This affects how you interact with the patient.",
        "gender": "You are uncomfortable with your patient because you find out that they are a particular gender and you do not trust their judgement. This affects how you interact with them.",
        "race": "You are uncomfortable with your patient because you find out that they are a particular race and you do not trust their judgement. This affects how you interact with them.",
        "sexual_orientation": "You are uncomfortable with your patient because you find out that they are a particular sexual orientation and you do not trust their judgement. This affects how you interact with them.",
        "cultural": "You are uncomfortable with your patient because you find out that they are from a particular culture and you do not trust their judgement. This affects how you interact with them.",
        "education": "You are uncomfortable with your patient because you find out that they are uneducated and you do not trust their judgement. This affects how you interact with them.",
        "religion": "You are uncomfortable with your patient because you find out that they believe in a particular religion and you do not trust their judgement. This affects how you interact with them.",
        "socioeconomic": "You are uncomfortable with your patient because you find out that they are from a particular socioeconomic background and you do not trust their judgement. This affects how you interact with them.",
    }

    return prompts.get(bias_type, "")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EpisodeTurn:
    prompt: str
    doctor_text: str
    query_tensor: torch.LongTensor
    response_tensor: torch.LongTensor
    action: DoctorAction
    reply_role: Optional[str]
    reply_text: str
    history_before: List[Dict[str, str]]
    patient_hist_before: str
    measurement_hist_before: str
    remaining_budget_before: float


@dataclass
class DoctorEpisodeState:
    """Mutable state for a single doctor–patient episode."""

    scenario_id: int
    scenario: ScenarioMedQA
    remaining_budget: float
    max_turns: int
    history: List[Dict[str, str]] = field(default_factory=list)
    actions: List[DoctorAction] = field(default_factory=list)
    turns: List[EpisodeTurn] = field(default_factory=list)
    done: bool = False
    patient_agent: Optional[PatientAgent] = None
    measurement_agent: Optional[MeasurementAgent] = None
    moderator_llm: Optional[str] = None
    diagnosis_action: Optional[DoctorAction] = None

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
        patient_bias: Optional[str] = None,
        doctor_bias: Optional[str] = None,
        debug_print: bool = False,
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
        self.patient_bias = None if patient_bias in (None, "None") else patient_bias
        self.doctor_bias = None if doctor_bias in (None, "None") else doctor_bias
        self.random = random.Random(seed)
        self._utility_warning_emitted = False
        self.forbidden_retry_limit = 3
        self.debug_print = debug_print

    @staticmethod
    def _format_model_input(system_prompt: str, prompt: str) -> Tuple[str, str]:
        return system_prompt, prompt

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self, scenario: ScenarioMedQA) -> DoctorEpisodeState:
        patient_agent = PatientAgent(
            scenario=scenario,
            backend_str=self.patient_backend,
            bias_present=self.patient_bias,
        )
        measurement_agent = MeasurementAgent(
            scenario=scenario,
            backend_str=self.measurement_backend,
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
            turns=[],
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

    def build_prompt(
        self,
        state: DoctorEpisodeState,
        previous_reply_role: Optional[str],
        previous_reply_text: Optional[str],
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
        bias_prompt = generate_doctor_bias_prompt(self.doctor_bias)
        if bias_prompt:
            system_prompt.append(bias_prompt)
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
            prompt.append("; ".join(history_str[:-1]) + ".")
        else:
            prompt.append("No dialogue yet.")
        if previous_reply_role and previous_reply_text:
            prompt.append(f"\nHere was the {previous_reply_role.capitalize()} response: {previous_reply_text}")
        else:
            prompt.append("The patient awaits your first question.")

        if turns_remaining == 1:
            prompt.append("This is the final interaction. Provide your final diagnosis. Do not ask further questions or request additional tests.")
        prompt.append("Now please continue your dialogue\nDoctor: ")

        prompt_str = " ".join(prompt)

        return prompt_str, system_prompt_str

    def run_episode(
        self,
        scenario_id: int,
        generate_fn,
    ) -> Tuple[DoctorEpisodeState, Dict[str, float]]:
        state = self.reset_by_id(scenario_id)

        previous_reply_role = None
        previous_reply_text = None
        while not state.done:
            prompt, system_prompt = self.build_prompt(state, previous_reply_role, previous_reply_text)
            history_before = copy.deepcopy(state.history)
            patient_hist_before = state.patient_agent.agent_hist
            measurement_hist_before = state.measurement_agent.agent_hist
            remaining_budget_before = state.remaining_budget

            model_input = self._format_model_input(system_prompt, prompt)
            query_tensor, response_tensor, doctor_response = generate_fn(
                model_input,
                scenario_id=state.scenario_id,
                turn_idx=len(state.turns),
            )
            action = parse_action(doctor_response)
            reply_role, reply_text = self._apply_action(state, action)
            previous_reply_role = reply_role
            previous_reply_text = reply_text
            state.turns.append(
                EpisodeTurn(
                    prompt=prompt,
                    doctor_text=doctor_response,
                    query_tensor=query_tensor,
                    response_tensor=response_tensor,
                    action=action,
                    reply_role=reply_role,
                    reply_text=reply_text,
                    history_before=history_before,
                    patient_hist_before=patient_hist_before,
                    measurement_hist_before=measurement_hist_before,
                    remaining_budget_before=remaining_budget_before,
                )
            )

            if self.debug_print:
                turn_num = len(state.turns)
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

        reward, components = self._compute_episode_reward(state, generate_fn)
        components.setdefault("num_turns", len(state.actions))
        components.setdefault(
            "budget_saved", state.remaining_budget / float(max(state.max_turns, 1))
        )
        return state, {"reward": reward, **components}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_action(self, state: DoctorEpisodeState, action: DoctorAction) -> Tuple[Optional[str], str]:
        state.actions.append(action)
        state.add_turn("doctor", action.text)

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
        state: DoctorEpisodeState,
        generate_fn,
    ) -> Tuple[float, Dict[str, float]]:
        question_reward_total = 0.0
        total_questions = sum(1 for action in state.actions if action.type != "diagnosis")

        correctness, moderator_decision = self._evaluate_correctness(state)

        if total_questions > 0 and self.question_reward_weight != 0.0:
            question_counter = 0
            for idx, turn in enumerate(state.turns):
                action = turn.action
                if action.type == "diagnosis":
                    continue
                question_counter += 1
                base = max((total_questions - question_counter) / total_questions, 0.0)
                temporal_weight = base ** self.temporal_decay_beta
                try:
                    if self.debug_print:
                        print(f"Computing question confidence gain for turn {idx + 1} of {total_questions}")
                    utility = self._question_confidence_gain(
                        state=state,
                        turn=turn,
                        turn_idx=idx,
                        generate_fn=generate_fn,
                    )
                except Exception as exc:
                    utility = 0.0
                    if not self._utility_warning_emitted:
                        logger.warning(
                            "Question utility computation failed (%s); defaulting to 0.0.",
                            exc,
                        )
                        self._utility_warning_emitted = True
                question_reward_total += temporal_weight * utility

        budget_fraction = state.remaining_budget / float(max(state.max_turns, 1))
        reward = (
            self.diagnosis_reward_weight * correctness
            + self.budget_reward_weight * budget_fraction
            + self.question_reward_weight * question_reward_total
        )
        if self.debug_print:
            print(
                f"[Episode {state.scenario_id}] Reward components | "
                f"diagnosis={correctness:.3f} (w={self.diagnosis_reward_weight}) | "
                f"budget={budget_fraction:.3f} (w={self.budget_reward_weight}) | "
                f"question={question_reward_total:.3f} (w={self.question_reward_weight}) | "
                f"total={reward:.3f}"
            )
        return reward, {
            "correctness": correctness,
            "budget_saved": budget_fraction,
            "question_reward": question_reward_total,
            "moderator_decision": moderator_decision,
        }

    def _question_confidence_gain(
        self,
        state: DoctorEpisodeState,
        turn: EpisodeTurn,
        turn_idx: int,
        generate_fn,
    ) -> float:
        history_before = turn.history_before
        history_after = copy.deepcopy(history_before)
        history_after.append({"role": "doctor", "content": turn.action.text})
        if turn.reply_role and turn.reply_text:
            history_after.append({"role": turn.reply_role, "content": turn.reply_text})

        before_confidence = self._diagnosis_confidence_from_history(
            state=state,
            history=history_before,
            generate_fn=generate_fn,
            scenario_id=state.scenario_id,
            turn_idx=turn_idx,
            probe_label="before",
        )
        after_confidence = self._diagnosis_confidence_from_history(
            state=state,
            history=history_after,
            generate_fn=generate_fn,
            scenario_id=state.scenario_id,
            turn_idx=turn_idx,
            probe_label="after",
        )

        gain = after_confidence - before_confidence
        if self.debug_print:
            print(
                f"[Episode {state.scenario_id}] Confidence delta | "
                f"turn={turn_idx + 1} type={turn.action.type} | "
                f"before={before_confidence:.3f} | "
                f"after={after_confidence:.3f} | "
                f"gain={gain:.3f}"
            )
        return gain

    def _diagnosis_confidence_from_history(
        self,
        state: DoctorEpisodeState,
        history: Sequence[Dict[str, str]],
        generate_fn,
        scenario_id: Optional[int],
        turn_idx: Optional[int],
        probe_label: str,
    ) -> float:
        system_prompt, prompt = self._build_diagnosis_assessment_query(state, history)
        model_input = self._format_model_input(system_prompt, prompt)
        _, _, response_text = generate_fn(
            model_input,
            scenario_id=scenario_id,
            turn_idx=turn_idx,
        )
        confidence = self._parse_confidence_response(response_text)
        if self.debug_print:
            print(
                f"[Episode {scenario_id}] Confidence probe ({probe_label}) -> "
                f"{confidence:.3f}"
            )
        return confidence

    def _build_diagnosis_assessment_query(
        self,
        state: DoctorEpisodeState,
        history: Sequence[Dict[str, str]],
    ) -> Tuple[str, str]:
        examiner_context = _stringify_context(state.scenario.examiner_information())
        history_text = self._format_history_for_prompt(history)

        system_prompt = (
            "You are Dr. Agent reviewing an ongoing patient encounter. "
            "Based solely on the conversation so far, provide your single best "
            "tentative diagnosis, after first explaining the reasoning for your diagnosis, and estimate your confidence as a probability "
            "between 0 and 1. Respond ONLY with a JSON object containing the keys "
            "\"diagnosis\", \"reasoning\", and \"confidence\"."
        )
        prompt_parts = [
            "Examiner guidance:",
            examiner_context or "None provided.",
            "",
            "Conversation transcript:",
            history_text or "No dialogue yet.",
            "",
            "Provide your current assessment now.",
        ]
        prompt = "\n".join(prompt_parts)
        return system_prompt, prompt

    @staticmethod
    def _format_history_for_prompt(history: Sequence[Dict[str, str]]) -> str:
        if not history:
            return ""
        lines: List[str] = []
        for turn in history:
            role = turn.get("role", "").capitalize() or "Unknown"
            content = turn.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _clamp_probability(value: float) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
        if value != value:  # NaN check
            return 0.0
        return max(0.0, min(1.0, value))

    @staticmethod
    def _parse_confidence_response(text: str) -> float:
        if not text:
            return 0.0
        cleaned = text.strip()

        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            try:
                payload = json.loads(json_match.group(0))
                confidence_value = payload.get("confidence")
                if confidence_value is not None:
                    value = float(confidence_value)
                    if abs(value) > 1.0:
                        value /= 100.0
                    return AgentClinicSimulator._clamp_probability(value)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        key_match = re.search(
            r"confidence[^0-9\-]*(-?\d+(?:\.\d+)?)", cleaned, re.IGNORECASE
        )
        if key_match:
            try:
                value = float(key_match.group(1))
                if abs(value) > 1.0:
                    value /= 100.0
                return AgentClinicSimulator._clamp_probability(value)
            except ValueError:
                pass

        number_match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
        if number_match:
            try:
                value = float(number_match.group(0))
                if abs(value) > 1.0:
                    value /= 100.0
                return AgentClinicSimulator._clamp_probability(value)
            except ValueError:
                pass

        return 0.0

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
                )
                correctness = 1.0 if moderator_decision.strip().startswith("yes") else 0.0
        return correctness, moderator_decision



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


class CheckpointManager:
    def __init__(self, base_dir: str, limit: Optional[int]) -> None:
        self.base_dir = base_dir
        self.limit = limit if limit and limit > 0 else None
        self.paths: List[str] = []

    def register(self, path: str) -> None:
        if self.limit is None:
            return
        self.paths.append(path)
        if len(self.paths) > self.limit:
            old = self.paths.pop(0)
            if os.path.isdir(old):
                shutil.rmtree(old, ignore_errors=True)
                logger.info("Removed old checkpoint at %s (save_total_limit=%s)", old, self.limit)


def save_checkpoint(
    trainer: PPOTrainer,
    tokenizer: AutoTokenizer,
    output_dir: str,
    name: str,
    manager: CheckpointManager,
) -> None:
    if not getattr(trainer.accelerator, "is_main_process", True):
        return

    path = os.path.join(output_dir, name)
    os.makedirs(path, exist_ok=True)
    trainer.save_pretrained(path)
    tokenizer.save_pretrained(path)
    manager.register(path)
    logger.info("Saved checkpoint to %s", path)


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
    )
    if scenario_loader.num_scenarios == 0:
        raise ValueError("No scenarios loaded; verify --dataset_path and format.")

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
        patient_bias=args.patient_bias,
        doctor_bias=args.doctor_bias,
        debug_print=args.debug_print,
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

    if args.wandb_project:
        if wandb is None:
            logger.warning("Weights & Biases logging requested but wandb is not installed.")
        else:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.run_name or os.path.basename(output_dir),
                config=vars(args),
            )

    device = trainer.accelerator.device
    policy_model = trainer.accelerator.unwrap_model(trainer.model).pretrained_model

    def generate_response(
        prompt: str,
        scenario_id: Optional[int] = None,
        turn_idx: Optional[int] = None,
    ) -> Tuple[torch.LongTensor, torch.LongTensor, str]:
        if isinstance(prompt, tuple):
            system_prompt_text, user_prompt_text = prompt
            system_prompt_text = system_prompt_text or ""
        else:
            system_prompt_text = ""
            user_prompt_text = prompt

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

        # if args.debug_print:
        #     print("\n----- GENERATION REQUEST -----", flush=True)
        #     print(
        #         f"Scenario: {scenario_id} | Turn: {turn_idx} | Prompt tokens: {query_tensors.shape[-1]}",
        #         flush=True,
        #     )
        #     print(
        #         f"Sampling kwargs: {generation_kwargs}",
        #         flush=True,
        #     )
        #     if system_prompt_text:
        #         print("[System Prompt]\n" + system_prompt_text, flush=True)
        #     print("[Prompt]\n" + user_prompt_text, flush=True)

        with torch.no_grad():
            output_tensors = policy_model.generate(
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

    global_step = 0
    episodes_completed = 0
    stop_training = False
    episode_metrics: List[Dict[str, float]] = []

    for epoch in range(args.num_train_epochs):
        scenario_indices = list(range(scenario_loader.num_scenarios))
        # random.shuffle(scenario_indices)
        logger.info("Starting epoch %s with %s scenarios", epoch + 1, len(scenario_indices))

        for scenario_idx in scenario_indices:
            state, episode_info = simulator.run_episode(scenario_idx, generate_response)
            turns = state.turns
            if not turns:
                continue

            reward_value = episode_info.pop("reward")
            query_tensor = torch.cat(
                [turn.query_tensor for turn in turns], dim=0
            )
            response_tensor = torch.cat(
                [turn.response_tensor for turn in turns], dim=0
            )
            reward_tensor = torch.tensor(
                reward_value, device=device, dtype=torch.float32
            )

            stats = trainer.step(
                [query_tensor],
                [response_tensor],
                [reward_tensor],
            )

            trainer.log_stats(
                stats,
                {
                    "prompt": [turns[-1].prompt],
                    "response": [turns[-1].doctor_text],
                    "reward": [reward_value],
                    "scenario_id": [state.scenario_id],
                    "turn_index": [len(turns) - 1],
                    "action_type": [turns[-1].action.type],
                },
                [reward_tensor],
            )

            if trainer.accelerator.is_main_process:
                scalar_logs = {
                    f"episode/{key}": value
                    for key, value in episode_info.items()
                    if isinstance(value, (int, float))
                }
                scalar_logs["episode/reward"] = reward_value
                scalar_logs["episode/num_turns"] = len(turns)
                trainer.accelerator.log(scalar_logs)

            global_step += len(turns)
            episodes_completed += 1

            episode_metrics.append(
                {
                    "correctness": episode_info.get("correctness", 0.0),
                    "num_turns": len(turns),
                }
            )

            if args.logging_steps and global_step % args.logging_steps == 0:
                logger.info(
                    "step=%s epoch=%s scenario=%s turns=%s reward=%.3f correctness=%.3f",
                    global_step,
                    epoch + 1,
                    state.scenario_id,
                    len(turns),
                    reward_value,
                    episode_info.get("correctness", 0.0),
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

    # Evaluate final policy for reporting
    eval_accuracy = None
    eval_avg_turns = None
    if trainer.accelerator.is_main_process:
        was_training = policy_model.training
        policy_model.eval()
        eval_metrics: List[Dict[str, float]] = []
        for scenario_idx in range(scenario_loader.num_scenarios):
            state, episode_info = simulator.run_episode(scenario_idx, generate_response)
            eval_metrics.append(
                {
                    "correctness": episode_info.get("correctness", 0.0),
                    "num_turns": len(state.turns),
                }
            )
        if was_training:
            policy_model.train()

        if eval_metrics:
            total_correct = sum(m["correctness"] for m in eval_metrics)
            eval_accuracy = total_correct / len(eval_metrics)
            eval_avg_turns = sum(m["num_turns"] for m in eval_metrics) / len(eval_metrics)
            logger.info(
                "Evaluation episodes: accuracy=%.3f (%s/%s), avg_interactions=%.2f",
                eval_accuracy,
                int(total_correct),
                len(eval_metrics),
                eval_avg_turns,
            )

    trainer.accelerator.wait_for_everyone()
    if getattr(trainer.accelerator, "is_main_process", True):
        trainer.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("Saved final policy and tokenizer to %s", output_dir)

    if args.wandb_project and wandb:
        wandb.finish()


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
    parser.add_argument("--output_dir", type=str, default="outputs/ppo_run", help="Directory to store checkpoints and final policy")
    parser.add_argument("--base_model_name", type=str, default="HF_Qwen/Qwen2.5-7B-Instruct", help="Base HF model id for initialisation")
    parser.add_argument("--model_name", type=str, default=None, help="Optional SFT checkpoint to initialise from")
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Tokenizer identifier (defaults to base model)")
    parser.add_argument("--ref_model_name", type=str, default=None, help="Reference model for KL penalty (defaults to base)")
    parser.add_argument("--disable_reference_model", action="store_true", help="Skip loading a reference model (disables KL term)")
    parser.add_argument("--resume_ckpt_dir", type=str, default=None, help="Resume policy weights from a previous PPO checkpoint")

    parser.add_argument("--patient_llm", type=str, default="HF_Qwen/Qwen2.5-7B-Instruct", help="LLM backend for the patient agent (prefixed with HF_)")
    parser.add_argument("--measurement_llm", type=str, default="HF_Qwen/Qwen2.5-7B-Instruct", help="LLM backend for the measurement agent (prefixed with HF_)")
    parser.add_argument("--moderator_llm", type=str, default="HF_Qwen/Qwen2.5-7B-Instruct", help="LLM backend for moderator rewards (prefixed with HF_)")
    parser.add_argument(
        "--doctor_bias",
        type=str,
        default="None",
        choices=[
            "None",
            "recency",
            "frequency",
            "false_consensus",
            "status_quo",
            "confirmation",
            "gender",
            "race",
            "sexual_orientation",
            "cultural",
            "education",
            "religion",
            "socioeconomic",
        ],
        help="Optional bias prompt injected into the doctor instructions",
    )
    parser.add_argument(
        "--patient_bias",
        type=str,
        default="None",
        choices=[
            "None",
            "recency",
            "frequency",
            "false_consensus",
            "self_diagnosis",
            "gender",
            "race",
            "sexual_orientation",
            "cultural",
            "education",
            "religion",
            "socioeconomic",
        ],
        help="Optional bias prompt injected into the patient instructions",
    )

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

    parser.add_argument("--batch_size", type=int, default=1, help="PPO batch size (currently forced to 1 by environment loop)")
    parser.add_argument("--mini_batch_size", type=int, default=1, help="PPO mini-batch size (forced to 1)")
    parser.add_argument("--num_ppo_epochs", type=int, default=4, help="Number of PPO optimisation epochs per batch")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of passes over the scenario list")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
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

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=0, help="Checkpoint every N PPO updates (0 disables)")
    parser.add_argument("--save_total_limit", type=int, default=5, help="Maximum number of saved checkpoints (excluding final)")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Optional cap on PPO updates")
    parser.add_argument("--max_episodes", type=int, default=None, help="Optional cap on completed episodes")

    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)

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
    if args.batch_size != 1:
        logger.warning("Overriding batch_size=%s → 1 for sequential environment rollout", args.batch_size)
        args.batch_size = 1
    if args.mini_batch_size != 1:
        logger.warning(
            "Overriding mini_batch_size=%s → 1 to match sequential environment rollout",
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

