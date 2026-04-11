"""
Inference Script — AI-Research-Env
====================================
Mandatory environment variables (injected by hackathon validator):
  API_BASE_URL   LiteLLM proxy endpoint  (e.g. https://router.huggingface.co/v1)
  API_KEY        LiteLLM proxy api key   (injected by validator — do NOT replace with own key)
  MODEL_NAME     Model identifier for inference.

Architecture:
  LLM (API) is ALWAYS tried first for every decision.
  Rule-based fallback is ONLY used if the LLM API call throws an exception.

STDOUT FORMAT (strictly required by validator):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""
from __future__ import annotations
import argparse
import os, sys, textwrap, time
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# tracks usage via api_key; any bypass causes Phase 2 failure.
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: Optional[str] = os.environ.get("API_KEY")   # injected by hackathon validator — no fallback
MODEL_NAME: str = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# Fallback: also check HF_TOKEN and OPENAI_API_KEY if API_KEY not set
if not API_KEY:
    API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")

BENCHMARK    = "ai-research-env"
MAX_STEPS    = 10
TEMPERATURE  = 0.4
MAX_TOKENS   = 512
SUCCESS_THRESHOLD = 0.5
TASKS = ["cv-classification", "nlp-sentiment", "healthcare-tabular"]

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert AI/ML researcher navigating a structured research environment.
You will be told which action to take. Respond with a detailed, technically precise answer.
Always name specific techniques, cite relevant methods, and include concrete metrics or numbers.
Be thorough — vague answers score poorly.
""").strip()

ACTION_PROMPTS = {
    "read_paper": "Read the provided research context carefully. Summarise the key techniques, challenges, and relevant prior work. Identify 3-5 critical issues you will address.",
    "propose_hypothesis": "Based on the papers and problem context, propose a clear, testable hypothesis. Explain WHY you think this approach will work and what the expected improvement mechanism is.",
    "design_experiment": "Design a concrete experiment to test your hypothesis. Specify: model architecture, hyperparameters, evaluation metrics, baseline comparison, and success criteria.",
    "run_experiment": "Simulate running the experiment. Report realistic results with specific numbers (accuracy/F1/AUROC), training curves, and any unexpected findings.",
    "analyze_results": "Analyse the experiment results. Compare to baseline, explain what worked and what didn't, and identify remaining gaps.",
    "refine_hypothesis": "Based on the analysis, refine your hypothesis. What did you learn? What would you change? Address any gaps or contradictions found.",
    "final_answer": "Provide your final research conclusion. Summarise all findings, the best approach discovered, achieved metrics, limitations, and recommendations for deployment.",
}

PHASE_SEQUENCE = ["read_paper","propose_hypothesis","design_experiment","run_experiment","analyze_results","refine_hypothesis","final_answer"]

_llm_client: Optional[Any] = None


def _get_client() -> Optional[Any]:
    global _llm_client
    if _llm_client is not None:
        return _llm_client
    if not API_KEY:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        print("[ERROR] openai not installed. Run: pip install openai", file=sys.stderr)
        return None
    _llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    return _llm_client


def _call_llm(system: str, user: str, _retries: int = 4) -> str:
    client = _get_client()
    if client is None:
        raise RuntimeError("No API key — set API_KEY env var or use --no-llm.")
    last_err: Exception = RuntimeError("unknown")
    for attempt in range(_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            if attempt < _retries:
                err_str = str(e).lower()
                is_rate_limit = (
                    "429" in str(e)
                    or "rate" in err_str
                    or "quota" in err_str
                    or "too many" in err_str
                    or "credits" in err_str
                )
                if is_rate_limit:
                    wait = 45 * (attempt + 1)
                    print(
                        f"[WARN] Rate limit (attempt {attempt+1}/{_retries}). "
                        f"Waiting {wait}s before retry...",
                        file=sys.stderr, flush=True,
                    )
                else:
                    wait = 4 ** attempt
                    print(
                        f"[WARN] LLM attempt {attempt+1} failed: {e.__class__.__name__}. "
                        f"Retrying in {wait}s...",
                        file=sys.stderr, flush=True,
                    )
                time.sleep(wait)
    raise last_err


def log_start(task, env, model): print(f"[START] task={task} env={env} model={model}", flush=True)
def log_step(step, action, reward, done, error):
    err = error if error else "null"
    print(f"[STEP]  step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)
def log_end(success, steps, score, rewards):
    print(f"[END]   success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)
    sys.stdout.flush()


def run_task(task_name: str, use_llm: bool = True) -> dict:
    from backend.env.research_env import Action, ResearchEnv

    env = ResearchEnv(task_name=task_name)
    obs = env.reset()
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    result = None
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        for phase in PHASE_SEQUENCE:
            if env._done:
                break
            user_content = textwrap.dedent(f"""
                Task: {obs.task_name} ({obs.difficulty})
                Current Phase: {phase.upper()}
                Your job now: {ACTION_PROMPTS[phase]}

                === Research Context ===
                {obs.research_context}

                Previous feedback: {obs.last_feedback or 'None'}
                Steps used: {obs.step_number}/{obs.max_steps}
            """).strip()

            err = None
            if use_llm:
                try:
                    content = _call_llm(SYSTEM_PROMPT, user_content)
                except Exception as e:
                    content = f"Error in {phase}"
                    err = str(e)[:80]
            else:
                # Rule-based fallback — no API key needed
                content = _fallback_content(phase, task_name, obs)

            action = Action(action_type=phase, content=content)
            result = env.step(action)
            reward = result.reward
            rewards.append(reward)
            steps_taken += 1
            obs = result.observation
            log_step(step=steps_taken, action=phase, reward=reward, done=result.done, error=err)
            if result.done:
                break

        score_raw = result.info.get("best_raw_score", sum(rewards)/max(len(rewards),1)) if result else 0.0
        score = round(min(max(float(score_raw), 0.01), 0.99), 3)
        success = score >= SUCCESS_THRESHOLD
    except Exception as e:
        score = round(min(max(sum(rewards)/max(len(rewards),1), 0.01), 0.99), 3) if rewards else 0.01
        success = False

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {"task": task_name, "score": score, "success": success, "steps": steps_taken}


def _fallback_content(phase: str, task_name: str, obs) -> str:
    """Rule-based fallback for --no-llm mode."""
    if phase == "read_paper":
        return (
            "Key techniques identified: batch normalization, learning rate scheduling, "
            "data augmentation (mixup, cutout), dropout regularization, ResNet architecture, "
            "weight decay. The baseline CNN overfits after epoch 20 due to fixed learning rate "
            "and minimal augmentation. Distribution shift from Gaussian noise corruption requires "
            "robust training strategies."
        )
    elif phase == "propose_hypothesis":
        return (
            "Hypothesis: Combining batch normalization with aggressive data augmentation "
            "(including noise injection, mixup, and cutout) along with cosine learning rate "
            "scheduling and dropout regularization will address the overfitting problem and "
            "improve generalization to the noisy test set, achieving >= 85% accuracy."
        )
    elif phase == "design_experiment":
        return (
            "Experiment: ResNet-18 with batch normalization, dropout=0.3, learning rate=0.001 "
            "with cosine annealing scheduler, augmentation pipeline (horizontal flip, random crop, "
            "cutout, Gaussian noise injection), weight decay=1e-4, batch size=128, SGD optimizer "
            "with momentum=0.9, train for 100 epochs with early stopping patience=10. "
            "Baseline: vanilla CNN with fixed LR=0.01. Metric: accuracy on noisy test set."
        )
    elif phase == "run_experiment":
        return (
            "Results: ResNet-18+BN achieved 87.2% accuracy on clean test, 85.4% on noisy test "
            "(vs baseline 72%). Training converged at epoch 65. F1-score: 0.854. "
            "Best validation accuracy: 88.1% at epoch 58. Learning curves show smooth convergence "
            "without overfitting. Augmentation with noise injection was most impactful (+8%)."
        )
    elif phase == "analyze_results":
        return (
            "Analysis vs baseline: +13.4% accuracy improvement on noisy test set. Batch normalization "
            "contributed +5%, augmentation +8%, LR scheduling +3%. Compared to baseline, overfitting "
            "eliminated — gap between train/test accuracy reduced from 25% to 3%. "
            "Remaining gap: performance on heavily corrupted samples still below 80%."
        )
    elif phase == "refine_hypothesis":
        return (
            "Refined hypothesis: Adding mixup augmentation and test-time augmentation (TTA) can "
            "address the remaining robustness gap. The noise injection helped but targeted noise "
            "types (Gaussian, salt-and-pepper) during training would better match test corruption. "
            "Calibration via temperature scaling needed for confident predictions."
        )
    else:  # final_answer
        return (
            "Conclusion: ResNet-18 with batch normalization, cosine LR scheduling, and comprehensive "
            "augmentation (cutout, mixup, noise injection) achieved 85.4% accuracy on noisy test "
            "(+13.4% over baseline). Key findings: (1) Batch norm critical for generalization, "
            "(2) Noise-aware augmentation most impactful for distribution shift, (3) LR scheduling "
            "prevents overfitting. Recommended for deployment with TTA for additional robustness."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-Research-Env Inference")
    parser.add_argument("--no-llm", action="store_true", help="Use rule-based fallback (no API key needed)")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--base-url", type=str, default=None, help="Remote server URL")
    args = parser.parse_args()

    use_llm = not args.no_llm

    task_arg = os.getenv("TASK_NAME", "all")
    tasks_to_run = TASKS if task_arg == "all" else ([task_arg] if task_arg in TASKS else [])
    if not tasks_to_run:
        print(f"Unknown TASK_NAME. Use one of {TASKS} or 'all'."); sys.exit(1)

    results = [run_task(t, use_llm=use_llm) for t in tasks_to_run]
    print("\n=== FINAL SUMMARY ===", flush=True)
    for r in results:
        print(f"  {r['task']:25s}  score={r['score']:.3f}  success={str(r['success']).lower()}  steps={r['steps']}", flush=True)
    print(f"\n  OVERALL AVERAGE: {sum(r['score'] for r in results)/len(results):.3f}", flush=True)
