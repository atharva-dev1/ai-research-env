"""
Inference Script — AI-Research-Env
====================================
Mandatory stdout format:
  [START] task=<task> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>

Env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN, TASK_NAME
"""
from __future__ import annotations
import os, sys, textwrap
from typing import List, Optional
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
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

def log_start(task, env, model): print(f"[START] task={task} env={env} model={model}", flush=True)
def log_step(step, action, reward, done, error):
    err = error if error else "null"
    print(f"[STEP]  step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)
def log_end(success, steps, score, rewards):
    print(f"[END]   success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)

PHASE_SEQUENCE = ["read_paper","propose_hypothesis","design_experiment","run_experiment","analyze_results","refine_hypothesis","final_answer"]

def run_task(task_name: str) -> dict:
    sys.path.insert(0, os.path.dirname(__file__))
    from backend.env.research_env import Action, ResearchEnv

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
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
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":user_content}],
                    temperature=TEMPERATURE, max_tokens=MAX_TOKENS, stream=False,
                )
                content = (completion.choices[0].message.content or "").strip()
                err = None
            except Exception as e:
                content = f"Error in {phase}"
                err = str(e)[:80]

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
        score = round(min(max(float(score_raw), 0.0), 1.0), 3)
        success = score >= SUCCESS_THRESHOLD
    except Exception as e:
        score = round(sum(rewards)/max(len(rewards),1), 3) if rewards else 0.0
        success = False

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {"task": task_name, "score": score, "success": success, "steps": steps_taken}

if __name__ == "__main__":
    task_arg = os.getenv("TASK_NAME", "all")
    tasks_to_run = TASKS if task_arg == "all" else ([task_arg] if task_arg in TASKS else [])
    if not tasks_to_run:
        print(f"Unknown TASK_NAME. Use one of {TASKS} or 'all'."); sys.exit(1)

    results = [run_task(t) for t in tasks_to_run]
    print("\n=== FINAL SUMMARY ===", flush=True)
    for r in results:
        print(f"  {r['task']:25s}  score={r['score']:.3f}  success={str(r['success']).lower()}  steps={r['steps']}", flush=True)
    print(f"\n  OVERALL AVERAGE: {sum(r['score'] for r in results)/len(results):.3f}", flush=True)
