"""Tests for AI-Research-Env — run: pytest tests/ -v"""
import sys, os, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from backend.env.research_env import (
    Action, ResearchEnv, Observation, StateSnapshot,
    TASKS, VALID_ACTIONS, PHASE_ORDER, grade_action
)

TASK_NAMES = list(TASKS.keys())

# ── Spec compliance ────────────────────────────────────────────────────────

class TestSpecCompliance:
    @pytest.mark.parametrize("task", TASK_NAMES)
    def test_reset_returns_observation(self, task):
        env = ResearchEnv(task, seed=0)
        obs = env.reset()
        assert isinstance(obs, Observation)
        assert obs.task_name == task
        assert obs.step_number == 0
        assert len(obs.research_context) > 100
        assert len(obs.allowed_actions) > 0

    @pytest.mark.parametrize("task", TASK_NAMES)
    def test_step_types(self, task):
        env = ResearchEnv(task, seed=0)
        env.reset()
        result = env.step(Action(action_type="read_paper", content="Test review of batch normalisation and residual networks."))
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)
        assert isinstance(result.observation, Observation)
        assert isinstance(result.info, dict)

    @pytest.mark.parametrize("task", TASK_NAMES)
    def test_reward_in_range(self, task):
        env = ResearchEnv(task, seed=0)
        env.reset()
        for phase in ["read_paper", "propose_hypothesis", "design_experiment"]:
            if env._done: break
            r = env.step(Action(action_type=phase, content="Batch normalization overfitting augmentation learning rate schedule noise robust domain shift leakage imputation fairness xgboost shap auroc f1 accuracy epoch bert fine-tune curriculum"))
            assert 0.0 <= r.reward <= 1.0, f"{task}/{phase}: reward {r.reward} out of range"

    @pytest.mark.parametrize("task", TASK_NAMES)
    def test_state_snapshot(self, task):
        env = ResearchEnv(task, seed=0)
        env.reset()
        snap = env.state()
        assert isinstance(snap, StateSnapshot)
        assert snap.task_name == task
        assert snap.step_number == 0

    @pytest.mark.parametrize("task", TASK_NAMES)
    def test_reset_clears_state(self, task):
        env = ResearchEnv(task, seed=0)
        env.reset()
        env.step(Action(action_type="read_paper", content="something"))
        env.reset()
        s = env.state()
        assert s.step_number == 0
        assert s.cumulative_reward == 0.0
        assert s.phases_completed == []

    def test_done_after_final_answer(self):
        env = ResearchEnv("cv-classification", seed=0)
        env.reset()
        # Run through phases to unlock final_answer
        for phase in ["read_paper","propose_hypothesis","design_experiment","run_experiment","analyze_results"]:
            env.step(Action(action_type=phase, content="batch norm overfitting augment learning rate schedule accuracy 85 epoch resnet residual dropout weight decay"))
        result = env.step(Action(action_type="final_answer", content="Final: batch norm augmentation resnet schedule achieves 85% accuracy regularization"))
        assert result.done is True

    def test_step_after_done_raises(self):
        env = ResearchEnv("cv-classification", seed=0)
        env.reset()
        env._done = True
        with pytest.raises(RuntimeError):
            env.step(Action(action_type="read_paper", content="test"))

    def test_invalid_action_raises(self):
        env = ResearchEnv("cv-classification", seed=0)
        env.reset()
        with pytest.raises(ValueError):
            env.step(Action(action_type="hack_system", content="test"))


# ── Grader quality ─────────────────────────────────────────────────────────

class TestGraderQuality:
    def test_better_answer_scores_higher(self):
        env = ResearchEnv("cv-classification", seed=0)
        env.reset(); snippet = env._task_cfg

        weak = grade_action(snippet, Action(action_type="read_paper", content="The code has some issues."), "read_paper", [])
        strong = grade_action(snippet, Action(action_type="read_paper",
            content="ResNet uses batch normalization and residual connections to improve generalization. "
                    "Overfitting after epoch 20 suggests need for learning rate schedule and dropout. "
                    "Data augmentation beyond horizontal flip (mixup, cutout) would reduce noise sensitivity. "
                    "Weight decay regularization and cosine annealing scheduler are standard solutions."),
            "read_paper", [])
        assert strong.value > weak.value

    def test_deterministic(self):
        env1 = ResearchEnv("nlp-sentiment", seed=0); env1.reset()
        env2 = ResearchEnv("nlp-sentiment", seed=0); env2.reset()
        a = Action(action_type="propose_hypothesis", content="Label noise memorization overfitting domain shift early stopping curriculum learning robust bert fine-tune")
        r1 = env1.step(a)
        r2 = env2.step(a)
        assert r1.reward == r2.reward

    def test_phase_progression_tracked(self):
        env = ResearchEnv("healthcare-tabular", seed=0)
        env.reset()
        phases_hit = []
        for phase in ["read_paper", "propose_hypothesis", "design_experiment"]:
            env.step(Action(action_type=phase, content="leakage imputation missing mice xgboost shap auroc fairness bias subgroup temporal calibration smote"))
            phases_hit.append(phase)
        snap = env.state()
        for p in phases_hit:
            assert p in snap.phases_completed

    def test_final_answer_needs_completeness(self):
        # Full episode vs just final_answer immediately — full episode should score better
        env_full = ResearchEnv("cv-classification", seed=0)
        env_full.reset()
        long_text = ("batch norm augment resnet schedule accuracy 85 epoch overfitting regulariz "
                     "noise learning rate weight decay dropout generalization residual "
                     "baseline improved results comparison metric evaluation training curves "
                     "dropout cutout mixup cosine annealing warmup validation test accuracy "
                     "batch normalization accelerates convergence reduces internal covariate shift "
                     "augmentation diversity reduces overfitting to specific pixel patterns")
        for phase in ["read_paper","propose_hypothesis","design_experiment","run_experiment","analyze_results"]:
            env_full.step(Action(action_type=phase, content=long_text))
        r_full = env_full.step(Action(action_type="final_answer", content=long_text))

        env_short = ResearchEnv("cv-classification", seed=0)
        env_short.reset()
        r_short = env_short.step(Action(action_type="final_answer", content=long_text))

        assert r_full.info["raw_score"] >= r_short.info["raw_score"]


# ── Difficulty progression ─────────────────────────────────────────────────

class TestDifficulty:
    def test_harder_tasks_have_more_steps(self):
        assert TASKS["cv-classification"]["max_steps"] < TASKS["nlp-sentiment"]["max_steps"]
        assert TASKS["nlp-sentiment"]["max_steps"] < TASKS["healthcare-tabular"]["max_steps"]

    def test_harder_tasks_need_more_keywords(self):
        # Healthcare has more required keywords than CV
        cv_kws = len(TASKS["cv-classification"]["ground_truth"]["key_techniques"])
        hc_kws = len(TASKS["healthcare-tabular"]["ground_truth"]["key_techniques"])
        assert hc_kws >= cv_kws

    @pytest.mark.parametrize("task", TASK_NAMES)
    def test_perfect_answer_scores_high(self, task):
        env = ResearchEnv(task, seed=0)
        env.reset()
        cfg = env._task_cfg
        kws = cfg["ground_truth"]["key_techniques"]
        filler = ("This technique significantly improves model performance and generalization. "
                  "The experimental results demonstrate clear improvement over baseline. "
                  "Statistical significance was confirmed across multiple runs with different seeds. ") * 4
        perfect = " ".join(kws) + " " + filler
        result = env.step(Action(action_type="read_paper", content=perfect))
        assert result.info["raw_score"] >= 0.70, f"{task}: expected ≥0.70, got {result.info['raw_score']}"


# ── HTTP server (optional, needs RUN_SERVER_TESTS=1) ───────────────────────

class TestHTTP:
    @pytest.fixture(autouse=True)
    def skip(self):
        if not os.getenv("RUN_SERVER_TESTS"):
            pytest.skip("Set RUN_SERVER_TESTS=1")

    def test_health(self):
        import httpx
        r = httpx.get("http://localhost:7860/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_full_cycle(self):
        import httpx
        r = httpx.post("http://localhost:7860/reset", json={"task_name": "cv-classification"})
        assert r.status_code == 200
        sid = r.json()["session_id"]
        r2 = httpx.post("http://localhost:7860/step", json={
            "session_id": sid,
            "action": {"action_type": "read_paper",
                       "content": "ResNet batch normalization overfitting learning rate schedule augmentation noise residual dropout"},
        })
        assert r2.status_code == 200
        assert 0.0 <= r2.json()["reward"] <= 1.0
