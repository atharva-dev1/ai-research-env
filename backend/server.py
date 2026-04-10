"""FastAPI server — OpenEnv HTTP API + serves built React frontend"""
from __future__ import annotations
import os, sys, uuid
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.env.research_env import Action, ResearchEnv, Observation, StateSnapshot, StepResult, TASKS, VALID_ACTIONS

app = FastAPI(title="AI-Research-Env", description="OpenEnv scientific discovery environment", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_sessions: Dict[str, ResearchEnv] = {}

def _get(sid: str) -> ResearchEnv:
    env = _sessions.get(sid)
    if not env:
        raise HTTPException(404, f"Session '{sid}' not found. Call /reset first.")
    return env

class ResetRequest(BaseModel):
    task_name: str = "cv-classification"
    session_id: Optional[str] = None
    seed: Optional[int] = None

class ResetResponse(BaseModel):
    session_id: str
    observation: Observation

class StepRequest(BaseModel):
    session_id: str
    action: Action

class StepResponse(BaseModel):
    session_id: str
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]

class StateResponse(BaseModel):
    session_id: str
    state: StateSnapshot

@app.get("/health")
def health():
    return {"status": "ok", "env": "AI-Research-Env", "tasks": list(TASKS.keys()), "actions": VALID_ACTIONS}

@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest):
    if req.task_name not in TASKS:
        raise HTTPException(400, f"task_name must be one of {list(TASKS.keys())}")
    sid = req.session_id or str(uuid.uuid4())
    env = ResearchEnv(task_name=req.task_name, seed=req.seed)
    obs = env.reset()
    _sessions[sid] = env
    return ResetResponse(session_id=sid, observation=obs)

@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    env = _get(req.session_id)
    try:
        result: StepResult = env.step(req.action)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(400, str(e))
    return StepResponse(session_id=req.session_id, observation=result.observation,
                        reward=result.reward, done=result.done, info=result.info)

@app.get("/state/{session_id}", response_model=StateResponse)
def state(session_id: str):
    return StateResponse(session_id=session_id, state=_get(session_id).state())

@app.get("/tasks")
def list_tasks():
    return {name: {"difficulty": cfg["difficulty"], "max_steps": cfg["max_steps"],
                   "phases": cfg["phases"]} for name, cfg in TASKS.items()}

# Serve built React frontend (after `npm run build`)
_static = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static):
    app.mount("/assets", StaticFiles(directory=os.path.join(_static, "assets")), name="assets")

    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(_static, "index.html"))

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        # API routes already handled above; everything else → SPA
        f = os.path.join(_static, full_path)
        if os.path.isfile(f):
            return FileResponse(f)
        return FileResponse(os.path.join(_static, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="0.0.0.0", port=int(os.getenv("PORT", 7860)), reload=False)
