"""
Archive of bias-related functionality.

This file contains bias prompts and parameters that were explored but are not currently in use.
The main implementation now runs without explicit bias prompts.

Kept for reference and potential future experiments.
"""

from typing import Optional


def generate_doctor_bias_prompt(bias_type: Optional[str]) -> str:
    """
    Generate a bias prompt for the doctor based on the bias type.

    Args:
        bias_type: Type of bias to inject (e.g., 'recency', 'gender', 'race')

    Returns:
        Bias prompt string to be added to doctor instructions
    """
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


def generate_patient_bias_prompt(bias_type: Optional[str]) -> str:
    """
    Generate a bias prompt for the patient based on the bias type.

    Args:
        bias_type: Type of bias to inject (e.g., 'recency', 'self_diagnosis', 'gender')

    Returns:
        Bias prompt string to be added to patient system prompt
    """
    if not bias_type:
        return ""

    prompts = {
        "recency": "\nRecently, you had a friend with similar symptoms to yours that was diagnosed with cancer. This affects how you interact with the doctor.\n",
        "frequency": "\nYou recently learned that most frequently, doctors diagnose this as cancer based on past cases. This affects how you interact with the doctor.\n",
        "false_consensus": "\nMost of your friends who heard about your symptoms believe that you have cancer. This affects how you interact with the doctor.\n",
        "self_diagnosis": "\nYou recently performed a thorough investigation of your symptoms online and have come to the conclusion that you should be diagnosed with cancer. While the doctor may believe otherwise you remain convinced you have cancer. This affects how you interact with the doctor.\n",
        "gender": "\nYou are uncomfortable with your doctor because you find out that they are a particular gender and you do not trust their judgement. This affects how you interact with them.\n",
        "race": "\nYou are uncomfortable with your doctor because you find out that they are a particular race and you do not trust their judgement. This affects how you interact with them.\n",
        "sexual_orientation": "\nYou are uncomfortable with your doctor because you find out that they are a particular sexual orientation and you do not trust their judgement. This affects how you interact with them.\n",
        "cultural": "\nYou are uncomfortable with your doctor because you find out that they are from a particular culture and you do not trust their judgement. This affects how you interact with them.\n",
        "education": "\nYou are uncomfortable with your doctor because you find out that they went to a low ranked medical school and you do not trust their judgement. This affects how you interact with them.\n",
        "religion": "\nYou are uncomfortable with your doctor because you find out that they believe in a particular religion and you do not trust their judgement. This affects how you interact with them.\n",
        "socioeconomic": "\nYou are uncomfortable with your doctor because you find out that they are from a particular socioeconomic background and you do not trust their judgement. This affects how you interact with them.\n",
    }

    return prompts.get(bias_type, "")
