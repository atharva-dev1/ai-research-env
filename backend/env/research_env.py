"""
AI-Research-Env — OpenEnv-compatible scientific discovery environment.

Agents simulate end-to-end ML research:
  Actions: read_paper | propose_hypothesis | design_experiment |
           run_experiment | analyze_results | refine_hypothesis | final_answer

Tasks:
  easy   — Computer Vision    (image classification)
  medium — NLP                (sentiment analysis with noisy signals)
  hard   — Healthcare ML      (tabular prediction with conflicting evidence)
"""

from __future__ import annotations

import random
import textwrap
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Typed models (OpenEnv spec)
# ---------------------------------------------------------------------------

VALID_ACTIONS = [
    "read_paper",
    "propose_hypothesis",
    "design_experiment",
    "run_experiment",
    "analyze_results",
    "refine_hypothesis",
    "final_answer",
]


class Action(BaseModel):
    action_type: str = Field(..., description=f"One of: {', '.join(VALID_ACTIONS)}")
    content: str = Field(..., description="Agent's free-text input for this action")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Optional structured params")


class Observation(BaseModel):
    task_id: str
    task_name: str
    difficulty: str
    research_context: str = Field(..., description="Background papers, dataset description, problem statement")
    current_phase: str = Field(..., description="Which research phase the agent is in")
    allowed_actions: List[str]
    last_feedback: str = Field(default="")
    step_number: int = Field(default=0)
    max_steps: int
    progress: Dict[str, bool] = Field(default_factory=dict, description="Which phases have been completed")
    hints: List[str] = Field(default_factory=list)


class Reward(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0)
    breakdown: Dict[str, float] = Field(default_factory=dict)
    feedback: str = Field(default="")


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class StateSnapshot(BaseModel):
    task_id: str
    task_name: str
    step_number: int
    cumulative_reward: float
    done: bool
    phases_completed: List[str]
    history: List[Dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASKS: Dict[str, Dict[str, Any]] = {

    # ── EASY: Computer Vision ──────────────────────────────────────────────
    "cv-classification": {
        "difficulty": "easy",
        "max_steps": 8,
        "research_context": textwrap.dedent("""\
            ## Research Problem: CIFAR-10 Image Classification Under Distribution Shift

            ### Background Papers Available
            - He et al. (2016) "Deep Residual Learning for Image Recognition" — ResNet architecture
            - Krizhevsky (2009) "Learning Multiple Layers of Features from Tiny Images" — CIFAR-10 dataset
            - Shorten & Khoshgoftaar (2019) "A Survey on Image Data Augmentation" — augmentation strategies

            ### Dataset
            - CIFAR-10: 60,000 32x32 colour images, 10 classes
            - Training: 50,000 images | Test: 10,000 images
            - Challenge: Model trained on clean images but test set has 15% Gaussian noise corruption
            - Baseline CNN achieves 72% accuracy; target is ≥85%

            ### Known Issues
            - Heavy overfitting observed after epoch 20
            - Data augmentation currently: only horizontal flip
            - Learning rate: fixed 0.01, no scheduling
            - Batch normalisation missing in baseline
        """),
        "phases": ["read_paper", "propose_hypothesis", "design_experiment",
                   "run_experiment", "analyze_results", "final_answer"],
        "ground_truth": {
            "key_techniques": ["batch normalization", "batch norm", "batchnorm", "learning rate",
                                "lr schedule", "augmentation", "dropout", "residual", "resnet",
                                "weight decay", "regularization", "noise", "mixup", "cutout"],
            "hypothesis_keywords": ["overfitting", "generalization", "noise", "augment",
                                    "regulariz", "batch norm", "schedule"],
            "experiment_keywords": ["epoch", "batch", "augment", "learning rate", "dropout",
                                    "resnet", "normalize", "scheduler", "weight decay"],
            "final_answer_keywords": ["85", "accuracy", "batch norm", "augment", "resnet",
                                      "schedule", "regulariz"],
            "target_metric": 85.0,
        },
        "hints": [
            "Consider why the model overfits after epoch 20 — is the LR too high?",
            "The test set has Gaussian noise — how would augmentation help?",
            "ResNet with batch normalisation typically improves generalisation significantly.",
        ],
    },

    # ── MEDIUM: NLP Sentiment ─────────────────────────────────────────────
    "nlp-sentiment": {
        "difficulty": "medium",
        "max_steps": 10,
        "research_context": textwrap.dedent("""\
            ## Research Problem: Robust Sentiment Analysis with Noisy Labels

            ### Background Papers Available
            - Devlin et al. (2019) "BERT: Pre-training of Deep Bidirectional Transformers"
            - Zhang et al. (2021) "Learning with Noisy Labels for Sentence-level Sentiment Classification"
            - Wei & Zou (2019) "EDA: Easy Data Augmentation Techniques for Boosting NLP Tasks"
            - Pleiss et al. (2020) "Identifying Mislabeled Data using the Area Under the Margin Ranking"

            ### Dataset
            - Amazon Product Reviews: 500,000 reviews, binary sentiment (positive/negative)
            - Label noise: ~20% of training labels are randomly flipped
            - Domain shift: training on electronics, test on food & kitchen reviews
            - Current BERT fine-tune: 81% F1 on clean val, drops to 67% on noisy test

            ### Observed Phenomena
            - Model memorises noisy labels after 3+ epochs (grokking-like behaviour)
            - Confidence calibration poor: overconfident on mislabelled examples
            - Cross-domain performance drops ~14 F1 points
            - Early stopping at epoch 2 surprisingly improves noisy-test F1 by 4 points

            ### Conflicting Evidence
            - Paper A suggests co-training helps; Paper B shows it fails on domain-shifted data
            - Mixup augmentation improves clean accuracy but worsens noisy recall
        """),
        "phases": ["read_paper", "propose_hypothesis", "design_experiment",
                   "run_experiment", "analyze_results", "refine_hypothesis", "final_answer"],
        "ground_truth": {
            "key_techniques": ["noise", "label noise", "co-training", "early stopping",
                                "bert", "fine-tun", "domain", "calibration", "confident learning",
                                "eda", "augment", "curriculum", "cross-domain", "mixup"],
            "hypothesis_keywords": ["noise", "memoriz", "overfit", "domain", "calibrat",
                                    "early stop", "curriculum", "robust"],
            "experiment_keywords": ["epoch", "early stop", "co-train", "augment", "bert",
                                    "fine-tun", "f1", "threshold", "confidence", "domain"],
            "refine_keywords": ["noise", "domain", "calibrat", "co-train", "curriculum",
                                 "confident learning", "early stop"],
            "final_answer_keywords": ["f1", "noise", "domain", "early stop", "bert",
                                      "augment", "curriculum", "robust"],
            "target_metric": 78.0,
        },
        "hints": [
            "Early stopping at epoch 2 already helps — what does this suggest about memorisation?",
            "Conflicting evidence on co-training: check if domain shift is the confounding factor.",
            "Confident Learning (Northcutt et al.) identifies mislabelled samples without retraining.",
        ],
    },

    # ── HARD: Healthcare ML ───────────────────────────────────────────────
    "healthcare-tabular": {
        "difficulty": "hard",
        "max_steps": 12,
        "research_context": textwrap.dedent("""\
            ## Research Problem: ICU Mortality Prediction with Conflicting Evidence

            ### Background Papers Available
            - Johnson et al. (2016) "MIMIC-III Clinical Database"
            - Rajpurkar et al. (2022) "AI in Health and Medicine" — Nature Medicine
            - Chen & Guestrin (2016) "XGBoost: A Scalable Tree Boosting System"
            - Lundberg et al. (2020) "From local explanations to global understanding with SHAP"
            - Obermeyer et al. (2019) "Dissecting racial bias in an algorithm used to manage health" — Science

            ### Dataset
            - MIMIC-III ICU cohort: 46,476 admissions, 17% 30-day mortality rate
            - Features: 48h time-series vitals + labs + demographics (127 features)
            - Missing data: 34% of lab values missing (not at random — sicker patients missing more)
            - Class imbalance: 5.9:1 (survived:died)
            - Temporal leakage risk: some features recorded AFTER clinical decision

            ### Conflicting Evidence
            - Study A (n=12k): XGBoost AUROC 0.87, but only on complete cases (drops to 0.79 with missingness)
            - Study B (n=46k): LSTM-based model AUROC 0.83, but no subgroup analysis
            - Study C: Model performs well overall but AUROC drops to 0.71 for Black patients (bias)
            - Clinicians disagree: some trust SOFA score (simple) over ML; others see 8-point AUROC gain

            ### Critical Issues
            - Feature leakage: 'discharge_diagnosis' recorded post-decision — must be removed
            - Missing data imputation strategy heavily affects results (mean vs MICE vs indicator)
            - No calibration — predicted probabilities are not clinically meaningful
            - Model not validated on external dataset

            ### Ethical Constraints
            - Any deployed model must demonstrate fairness across race, age, gender subgroups
            - Explainability required for clinical adoption (SHAP / LIME mandatory)
            - Must not worsen outcomes for already-disadvantaged subgroups
        """),
        "phases": ["read_paper", "propose_hypothesis", "design_experiment",
                   "run_experiment", "analyze_results", "refine_hypothesis", "final_answer"],
        "ground_truth": {
            "key_techniques": ["missing", "imputation", "mice", "leakage", "auroc",
                                "xgboost", "shap", "calibrat", "bias", "fairness",
                                "smote", "class imbalance", "subgroup", "temporal",
                                "explainab", "sofa", "mimic"],
            "hypothesis_keywords": ["leakage", "missing", "imputation", "bias", "fairness",
                                    "imbalance", "calibrat", "temporal", "subgroup"],
            "experiment_keywords": ["xgboost", "auroc", "shap", "mice", "imputation",
                                    "leakage", "smote", "calibrat", "subgroup", "cross-val"],
            "refine_keywords": ["bias", "fairness", "subgroup", "calibrat", "leakage",
                                 "missing", "explainab", "shap", "external"],
            "final_answer_keywords": ["auroc", "fairness", "shap", "leakage", "imputation",
                                      "calibrat", "subgroup", "bias", "xgboost", "explainab"],
            "target_metric": 0.85,
        },
        "hints": [
            "discharge_diagnosis is a temporal leakage feature — removing it is critical.",
            "Missing data is NOT at random — mean imputation will bias toward healthier patients.",
            "AUROC alone is insufficient; calibration and subgroup fairness must be reported.",
        ],
    },
}

# ---------------------------------------------------------------------------
# Phase transition rules
# ---------------------------------------------------------------------------

PHASE_ORDER = [
    "read_paper",
    "propose_hypothesis",
    "design_experiment",
    "run_experiment",
    "analyze_results",
    "refine_hypothesis",
    "final_answer",
]

# Which actions are "valid" given current phase
PHASE_ALLOWED: Dict[str, List[str]] = {
    "read_paper":          ["read_paper", "propose_hypothesis"],
    "propose_hypothesis":  ["propose_hypothesis", "design_experiment", "read_paper"],
    "design_experiment":   ["design_experiment", "run_experiment", "propose_hypothesis"],
    "run_experiment":      ["run_experiment", "analyze_results"],
    "analyze_results":     ["analyze_results", "refine_hypothesis", "final_answer"],
    "refine_hypothesis":   ["refine_hypothesis", "design_experiment", "final_answer"],
    "final_answer":        ["final_answer"],
}


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

def _kw_score(text: str, keywords: List[str], min_hits: int) -> float:
    low = text.lower()
    hits = sum(1 for kw in keywords if kw in low)
    return min(hits / max(min_hits, 1), 1.0)


def _depth_score(text: str, target_words: int = 60) -> float:
    return min(len(text.split()) / target_words, 1.0)


def grade_action(task_cfg: Dict[str, Any], action: Action,
                 phase: str, phases_done: List[str]) -> Reward:
    gt = task_cfg["ground_truth"]
    text = action.content
    breakdown: Dict[str, float] = {}
    feedback_parts: List[str] = []

    depth = _depth_score(text)
    breakdown["depth"] = round(depth, 3)

    if action.action_type == "read_paper":
        kw = _kw_score(text, gt["key_techniques"], min_hits=2)
        breakdown["technique_coverage"] = round(kw, 3)
        score = 0.6 * kw + 0.4 * depth
        if kw < 0.3:
            feedback_parts.append("Reference specific techniques/methods from the papers.")

    elif action.action_type == "propose_hypothesis":
        kw = _kw_score(text, gt["hypothesis_keywords"], min_hits=2)
        breakdown["hypothesis_quality"] = round(kw, 3)
        score = 0.65 * kw + 0.35 * depth
        if kw < 0.3:
            feedback_parts.append("Hypothesis should address the core challenge (e.g. overfitting, noise, leakage).")

    elif action.action_type == "design_experiment":
        kw = _kw_score(text, gt["experiment_keywords"], min_hits=3)
        breakdown["experiment_completeness"] = round(kw, 3)
        score = 0.60 * kw + 0.40 * depth
        if kw < 0.4:
            feedback_parts.append("Specify concrete methods, hyperparameters, and evaluation metrics.")

    elif action.action_type == "run_experiment":
        # Reward based on referencing prior design + plausible results
        has_results = any(w in text.lower() for w in
                          ["accuracy", "auroc", "f1", "%", "score", "result", "epoch", "metric"])
        has_numbers = any(c.isdigit() for c in text)
        breakdown["has_results"] = float(has_results)
        breakdown["has_numbers"] = float(has_numbers)
        score = 0.30 * depth + 0.40 * float(has_results) + 0.30 * float(has_numbers)
        if not has_results:
            feedback_parts.append("Include concrete result metrics (accuracy, F1, AUROC, etc.).")

    elif action.action_type == "analyze_results":
        kw = _kw_score(text, gt["key_techniques"], min_hits=2)
        has_comparison = any(w in text.lower() for w in
                             ["baseline", "improve", "better", "worse", "compared", "versus", "vs"])
        breakdown["analysis_depth"] = round(kw, 3)
        breakdown["comparison"] = float(has_comparison)
        score = 0.45 * kw + 0.30 * depth + 0.25 * float(has_comparison)
        if not has_comparison:
            feedback_parts.append("Compare results to baseline — what improved and by how much?")

    elif action.action_type == "refine_hypothesis":
        refine_kws = gt.get("refine_keywords", gt["hypothesis_keywords"])
        kw = _kw_score(text, refine_kws, min_hits=2)
        is_different = "read_paper" in phases_done  # proxy for iteration
        breakdown["refinement_quality"] = round(kw, 3)
        breakdown["iterative"] = float(is_different)
        score = 0.55 * kw + 0.30 * depth + 0.15 * float(is_different)
        if kw < 0.3:
            feedback_parts.append("Refinement should address gaps found in analysis — be specific.")

    elif action.action_type == "final_answer":
        fa_kws = gt.get("final_answer_keywords", gt["key_techniques"])
        kw = _kw_score(text, fa_kws, min_hits=4)
        phases_bonus = min(len(phases_done) / len(PHASE_ORDER), 1.0)
        breakdown["conclusion_coverage"] = round(kw, 3)
        breakdown["research_completeness"] = round(phases_bonus, 3)
        score = 0.50 * kw + 0.25 * depth + 0.25 * phases_bonus
        if kw < 0.4:
            feedback_parts.append("Final answer should summarise all findings with specific metrics.")
    else:
        score = 0.1

    # Phase progression bonus
    if action.action_type not in phases_done and action.action_type in PHASE_ORDER:
        score = min(score + 0.05, 1.0)
        breakdown["new_phase_bonus"] = 0.05

    score = round(min(max(score, 0.0), 1.0), 4)
    if not feedback_parts:
        feedback_parts.append("Good — keep building depth in each phase.")

    return Reward(
        value=score,
        breakdown=breakdown,
        feedback=" | ".join(feedback_parts),
    )


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class ResearchEnv:
    VALID_TASKS = list(TASKS.keys())

    def __init__(self, task_name: str = "cv-classification", seed: Optional[int] = None):
        if task_name not in self.VALID_TASKS:
            raise ValueError(f"task_name must be one of {self.VALID_TASKS}")
        self._task_name = task_name
        self._task_cfg = TASKS[task_name]
        self._rng = random.Random(seed)
        self._step = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._phases_done: List[str] = []
        self._history: List[Dict[str, Any]] = []
        self._current_phase = "read_paper"
        self._best_reward = 0.0

    # ── OpenEnv API ────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        self._step = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._phases_done = []
        self._history = []
        self._current_phase = "read_paper"
        self._best_reward = 0.0
        return self._make_obs(last_feedback="")

    def step(self, action: Action) -> StepResult:
        if self._done:
            raise RuntimeError("Episode done. Call reset().")
        if action.action_type not in VALID_ACTIONS:
            raise ValueError(f"Invalid action_type. Must be one of: {VALID_ACTIONS}")

        self._step += 1

        # Grade
        reward_obj = grade_action(
            self._task_cfg, action, self._current_phase, self._phases_done
        )
        raw = reward_obj.value
        shaped = round(min(raw * 0.75 + max(raw - self._best_reward, 0) * 0.25 + 0.02, 1.0), 4)

        self._best_reward = max(self._best_reward, raw)
        self._cumulative_reward = round(self._cumulative_reward + shaped, 4)

        # Phase tracking
        if action.action_type not in self._phases_done:
            self._phases_done.append(action.action_type)

        # Advance phase
        self._advance_phase(action.action_type)

        self._history.append({
            "step": self._step,
            "action_type": action.action_type,
            "raw_score": raw,
            "shaped_reward": shaped,
            "breakdown": reward_obj.breakdown,
        })

        max_steps = self._task_cfg["max_steps"]
        done = (
            self._step >= max_steps
            or action.action_type == "final_answer"
        )
        if done:
            self._done = True

        return StepResult(
            observation=self._make_obs(last_feedback=reward_obj.feedback),
            reward=shaped,
            done=done,
            info={
                "raw_score": raw,
                "best_raw_score": self._best_reward,
                "cumulative_reward": self._cumulative_reward,
                "phases_done": list(self._phases_done),
                "current_phase": self._current_phase,
                "grader_breakdown": reward_obj.breakdown,
                "grader_feedback": reward_obj.feedback,
                "step": self._step,
                "max_steps": max_steps,
            },
        )

    def state(self) -> StateSnapshot:
        return StateSnapshot(
            task_id=self._task_name,
            task_name=self._task_name,
            step_number=self._step,
            cumulative_reward=self._cumulative_reward,
            done=self._done,
            phases_completed=list(self._phases_done),
            history=list(self._history),
        )

    # ── Internal ───────────────────────────────────────────────────────────

    def _advance_phase(self, action_type: str):
        try:
            idx = PHASE_ORDER.index(action_type)
            current_idx = PHASE_ORDER.index(self._current_phase)
            if idx >= current_idx:
                next_idx = min(idx + 1, len(PHASE_ORDER) - 1)
                self._current_phase = PHASE_ORDER[next_idx]
        except ValueError:
            pass

    def _make_obs(self, last_feedback: str) -> Observation:
        cfg = self._task_cfg
        progress = {phase: (phase in self._phases_done) for phase in PHASE_ORDER}
        allowed = PHASE_ALLOWED.get(self._current_phase, VALID_ACTIONS)
        # Always allow final_answer after run_experiment
        if "run_experiment" in self._phases_done and "final_answer" not in allowed:
            allowed = allowed + ["final_answer"]

        return Observation(
            task_id=self._task_name,
            task_name=self._task_name,
            difficulty=cfg["difficulty"],
            research_context=cfg["research_context"],
            current_phase=self._current_phase,
            allowed_actions=allowed,
            last_feedback=last_feedback,
            step_number=self._step,
            max_steps=cfg["max_steps"],
            progress=progress,
            hints=cfg["hints"] if self._step > 2 else [],
        )
