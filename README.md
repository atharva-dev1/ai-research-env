---
title: AI Research Env
emoji: 🔬
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
tags:
  - openenv
  - ai-research
  - scientific-discovery
  - rl-environment
  - reinforcement-learning
pinned: false
---

# 🔬 AI-Research-Env

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/atharvsha01/ai_research_env)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-green.svg)]()

**An OpenEnv-compatible simulation platform for end-to-end scientific discovery using AI agents.**

Instead of treating LLMs as simple prompt-response systems, this environment trains agents to operate inside a **structured research workflow** — reading papers, forming hypotheses, designing experiments, running them, analysing results, and producing final conclusions. This mirrors real ML research, making agents trained here directly useful in practice.

---

## 🧪 Why This Environment Is Not Trivial

Most agent benchmarks present isolated tasks. AI-Research-Env forces agents through a **7-phase scientific method** where each phase builds on the previous. The grader is **evidence-gated** — it checks for specific technical keywords, experimental rigour, and depth of analysis. A model that gives vague answers scores ~0.15; a model that demonstrates genuine ML research understanding scores ~0.75+.

The hard task (`healthcare-tabular`) specifically penalises common LLM failure modes:
- **Temporal leakage blindness**: The dataset contains a post-decision feature (`discharge_diagnosis`). Models that don't catch this lose critical points.
- **Missing data naivety**: 34% missing values are NOT at random — mean imputation biases toward healthier patients.
- **Fairness ignorance**: AUROC drops to 0.71 for Black patients. Models must address subgroup fairness.
- **Conflicting evidence**: Three studies give contradictory results — the agent must reason through them.

---

## 🧠 Agent Actions

Agents interact through 7 actions that simulate real ML research workflows:

| Action | Description |
|--------|-------------|
| `read_paper` | Summarise relevant literature and identify key challenges |
| `propose_hypothesis` | Form a testable hypothesis grounded in the literature |
| `design_experiment` | Specify model, hyperparameters, metrics, and baselines |
| `run_experiment` | Report simulated results with concrete numbers |
| `analyze_results` | Compare to baseline, identify gaps, explain findings |
| `refine_hypothesis` | Iterate based on evidence — address contradictions |
| `final_answer` | Deliver a complete research conclusion with recommendations |

---

## 📋 Tasks (Easy → Hard)

| Task | Difficulty | Max Steps | Domain |
|------|-----------|-----------|--------|
| `cv-classification` | 🟢 Easy | 8 | CIFAR-10 under distribution shift — fix overfitting + noise |
| `nlp-sentiment` | 🟡 Medium | 10 | Sentiment analysis with 20% noisy labels + domain shift |
| `healthcare-tabular` | 🔴 Hard | 12 | ICU mortality prediction — leakage, missingness, fairness |

Each task provides:
- Detailed research context with simulated papers, dataset stats, and conflicting evidence
- Phase-aware grading (keyword coverage + depth + progression)
- Partial-progress reward shaping across the full episode
- Contextual hints unlocked after step 2

---

## 🔌 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check + task/action list |
| `POST` | `/reset` | Start new episode (no body required) |
| `POST` | `/step` | Submit one action |
| `GET` | `/state?session_id=...` | Full state snapshot |
| `GET` | `/tasks` | List tasks with metadata |
| `GET` | `/metadata` | Environment metadata |
| `GET` | `/schema` | Action/observation/state schemas |
| `GET` | `/openenv.yaml` | OpenEnv spec file |
| `POST` | `/mcp` | JSON-RPC 2.0 MCP endpoint |
| `GET` | `/docs` | Interactive Swagger UI |

### Example

```bash
# Start episode (no body needed)
curl -X POST https://atharvsha01-ai-research-env.hf.space/reset

# Submit action
curl -X POST https://atharvsha01-ai-research-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "<from reset>",
    "action_type": "read_paper",
    "content": "ResNet uses batch normalisation to combat overfitting. Adding LR scheduling and stronger augmentation (cutout, mixup) should help the 15% noise corruption on the test set."
  }'
```

---

## 📐 Observation / Action Spaces

### Observation
```json
{
  "task_id": "cv-classification",
  "task_name": "cv-classification",
  "difficulty": "easy",
  "research_context": "## Research Problem: CIFAR-10...",
  "current_phase": "propose_hypothesis",
  "allowed_actions": ["propose_hypothesis", "design_experiment", "read_paper"],
  "last_feedback": "Good coverage of batch norm! Add specifics on LR scheduling.",
  "step_number": 2,
  "max_steps": 8,
  "progress": {"read_paper": true, "propose_hypothesis": false},
  "hints": ["Consider why the model overfits after epoch 20..."]
}
```

### Action
```json
{
  "action_type": "propose_hypothesis",
  "content": "Hypothesis: adding batch normalisation + cosine LR schedule + cutout augmentation will reduce overfitting and improve noise robustness from 72% to ≥85% accuracy.",
  "parameters": null
}
```

### Reward
- Range: **0.0 – 1.0** per step (shaped, not sparse)
- Components: keyword coverage (50–65%) + depth (25–35%) + phase progression bonus (5%)
- Episode reward: sum over all steps

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Rule-based fallback — no API key, fully deterministic, verifies env works
python inference.py --no-llm

# LLM agent via Hugging Face Inference Router
export API_BASE_URL="https://router.huggingface.co/v1"
export API_KEY="your_api_key_here"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
python inference.py

# Start the API server locally
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run inference against the running server
python inference.py --base-url http://localhost:7860
```

## Docker

```bash
# Build — runs reset()→step() validation at build time (fails fast if broken)
docker build -t ai-research-env .

# Run
docker run -p 7860:7860 \
  -e API_BASE_URL="https://router.huggingface.co/v1" \
  -e API_KEY="your_api_key" \
  -e MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" \
  ai-research-env

# Verify
curl http://localhost:7860/health
curl http://localhost:7860/tasks
curl -X POST http://localhost:7860/reset
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_KEY` | Yes* | — | LiteLLM proxy API key (injected by hackathon validator) |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | OpenAI-compatible LLM endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model to use |
| `HF_TOKEN` | No | — | Hugging Face API key (fallback for API_KEY) |
| `TASK_NAME` | No | `all` | Task to run in inference |

*Not required when using `--no-llm` mode.

---

## 📊 Baseline Scores

Fully reproducible with no API key:
```bash
python inference.py --no-llm
```

| Task | Score | Steps | Success |
|------|-------|-------|---------|
| `cv-classification` | ~0.985 | 7 | ✅ |
| `nlp-sentiment` | ~0.983 | 7 | ✅ |
| `healthcare-tabular` | ~0.970 | 7 | ✅ |
| **Average** | **~0.979** | — | — |

The rule-based fallback is deterministic — same input, same output, every run.

LLM baseline expected range (`Qwen/Qwen2.5-72B-Instruct`): **0.68 – 0.80** average across all tasks.

---

## 🗂️ Project Structure

```
ai-research-env/
├── server/
│   ├── __init__.py
│   └── app.py                 ← OpenEnv entry point (re-exports backend.server)
├── backend/
│   ├── env/
│   │   └── research_env.py    ← Core environment + grader (Pydantic typed)
│   ├── server.py              ← FastAPI HTTP server + serves React SPA
│   └── static/                ← Built React frontend (generated by npm run build)
├── frontend/
│   ├── src/
│   │   ├── App.jsx            ← Main dashboard (React + Recharts)
│   │   └── store/useStore.js  ← Zustand state management
│   ├── vite.config.js
│   └── package.json
├── tests/
│   └── test_server.py         ← Server endpoint validation
├── inference.py               ← Baseline script: LLM mode + rule-based fallback (--no-llm)
├── openenv.yaml               ← OpenEnv spec: tasks, action/obs spaces, env_vars, tags
├── pyproject.toml             ← Project metadata + dependencies
├── uv.lock                    ← Locked dependency versions
├── requirements.txt
├── Dockerfile                 ← Multi-stage: builds frontend then serves from Python
└── README.md                  ← This file (also HF Space card)
```

---

## ✅ OpenEnv Spec Compliance

- ✅ Typed Pydantic `Action`, `Observation`, `State`, `StepResult` models
- ✅ `reset()` → clean initial observation (no shared state between episodes)
- ✅ `step(action)` → `(observation, reward, done, info)` tuple
- ✅ `state()` → episode metadata, does not advance episode
- ✅ `openenv.yaml` — name, version, tasks, action space, observation space, env_vars, tags
- ✅ 3 tasks with difficulty gradient: easy → medium → hard
- ✅ Graders: deterministic, scores in 0.0–1.0, reproducible, variance across model quality
- ✅ `inference.py` in root, uses `from openai import OpenAI` (lazy import — safe without openai installed)
- ✅ Reads `API_BASE_URL`, `MODEL_NAME`, `API_KEY` / `HF_TOKEN` from environment
- ✅ Dockerfile: EXPOSE 7860, HEALTHCHECK, PYTHONPATH=/app, ENV defaults, build-time validation
- ✅ HF Space: `sdk: docker`, `app_port: 7860`, `tags: [openenv, …]`
- ✅ `/health` returns 200 (HF Space liveness probe)
- ✅ `server/app.py` present for multi-mode deployment
- ✅ Baseline scores reproducible: `python inference.py --no-llm`

---

## 🏆 Evaluation Criteria Alignment

| Criterion | Weight | How we address it |
|-----------|--------|------------------|
| Real-world utility | 30% | Scientific discovery is core ML workflow — trained agents improve real research pipelines |
| Task & grader quality | 25% | 3 tasks easy→hard, deterministic graders, scores [0,1], hard task includes fairness/leakage/missingness |
| Environment design | 20% | Phase-aware state, partial-progress reward, sensible episode boundaries, typed Pydantic models |
| Code quality & spec | 15% | OpenEnv spec, Dockerfile, openenv.yaml, server tests, documented |
| Creativity & novelty | 10% | 7-action research workflow is novel; reward combines keyword coverage + depth + phase progression |

---

## Author
Developed by **Atharva** ([@atharva-dev1](https://github.com/atharva-dev1))

---

## License
MIT
