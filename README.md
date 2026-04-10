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

**An OpenEnv-compatible simulation platform for end-to-end scientific discovery using AI agents.**

Instead of treating LLMs as simple prompt-response systems, this environment trains agents to operate inside a **structured research workflow** — reading papers, forming hypotheses, designing experiments, running them, analysing results, and producing final conclusions. This mirrors real ML research, making agents trained here directly useful in practice.

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

| Task ID | Difficulty | Max Steps | Domain |
|---------|-----------|-----------|--------|
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
| `POST` | `/reset` | Start new episode |
| `POST` | `/step` | Submit one action |
| `GET` | `/state/{session_id}` | Full state snapshot |
| `GET` | `/tasks` | List tasks with metadata |
| `GET` | `/docs` | Interactive Swagger UI |

### Example

```bash
# Start episode
curl -X POST https://atharvsha01-ai-research-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "cv-classification"}'

# Submit action
curl -X POST https://atharvsha01-ai-research-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "<from reset>",
    "action": {
      "action_type": "read_paper",
      "content": "ResNet uses batch normalisation to combat overfitting. Adding LR scheduling and stronger augmentation (cutout, mixup) should help the 15% noise corruption on the test set."
    }
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

## 🚀 Setup & Usage

### Run locally (Docker)

```bash
git clone https://huggingface.co/spaces/atharvsha01/ai_research_env
cd ai_research_env

docker build -t ai-research-env .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_hf_token \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  ai-research-env
```

Then open **http://localhost:7860** for the React dashboard, or **http://localhost:7860/docs** for the API.

### Run locally (dev mode — no Docker)

```bash
# Terminal 1: backend
pip install -r requirements.txt
python -m uvicorn backend.server:app --port 7860 --reload

# Terminal 2: frontend
cd frontend
npm install
npm run dev   # → http://localhost:5173 (proxies API to :7860)
```

### Run baseline inference

```bash
export HF_TOKEN=your_hf_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export TASK_NAME=all   # or: cv-classification | nlp-sentiment | healthcare-tabular

pip install -r requirements.txt
python inference.py
```

### Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes | — | Hugging Face API key |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model to use |
| `TASK_NAME` | No | `all` | Task to run in inference |

---

## 📊 Baseline Scores

Measured with `Qwen/Qwen2.5-72B-Instruct`:

| Task | Score | Steps | Success |
|------|-------|-------|---------|
| `cv-classification` | ~0.74 | 6 | ✅ |
| `nlp-sentiment` | ~0.68 | 7 | ✅ |
| `healthcare-tabular` | ~0.61 | 8 | ✅ |
| **Average** | **~0.68** | — | — |

*(Run `python inference.py` to reproduce)*

---

## 🗂️ Project Structure

```
ai-research-env/
├── backend/
│   ├── env/
│   │   └── research_env.py     ← Core environment + grader (Pydantic typed)
│   ├── server.py               ← FastAPI HTTP server + serves React SPA
│   └── static/                 ← Built React frontend (generated by npm run build)
├── frontend/
│   ├── src/
│   │   ├── App.jsx             ← Main dashboard (React + Recharts)
│   │   └── store/useStore.js   ← Zustand state management
│   ├── vite.config.js
│   └── package.json
├── tests/
│   └── test_env.py             ← 27 passing tests
├── inference.py                ← Mandatory baseline script
├── openenv.yaml                ← OpenEnv spec metadata
├── requirements.txt
├── Dockerfile                  ← Multi-stage: builds frontend then serves from Python
└── README.md
```

---

## 🏆 Evaluation Criteria Alignment

| Criterion | Weight | How we address it |
|-----------|--------|------------------|
| Real-world utility | 30% | Scientific discovery is core ML workflow — trained agents improve real research pipelines |
| Task & grader quality | 25% | 3 tasks easy→hard, deterministic graders, scores [0,1], hard task includes fairness/leakage/missingness |
| Environment design | 20% | Phase-aware state, partial-progress reward, sensible episode boundaries, typed Pydantic models |
| Code quality & spec | 15% | OpenEnv spec, Dockerfile, openenv.yaml, 27 tests passing, documented |
| Creativity & novelty | 10% | 7-action research workflow is novel; reward combines keyword coverage + depth + phase progression |

---

## Author
**Atharva** ([@atharva-dev1](https://github.com/atharva-dev1))

---

## License
MIT
