"""FastAPI server — OpenEnv HTTP API + serves built React frontend"""
from __future__ import annotations
import os, sys, uuid
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.env.research_env import Action, ResearchEnv, Observation, StateSnapshot, StepResult, TASKS, VALID_ACTIONS

app = FastAPI(title="AI-Research-Env", description="OpenEnv scientific discovery environment", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

_sessions: Dict[str, ResearchEnv] = {}

def _get(sid: str) -> ResearchEnv:
    env = _sessions.get(sid)
    if not env:
        raise HTTPException(404, f"Session '{sid}' not found. Call /reset first.")
    return env


@app.get("/")
def root() -> Dict[str, str]:
    """Root endpoint — HF Spaces health probe hits this first."""
    return {"status": "healthy", "environment": "ai-research-env", "version": "1.0.0", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy", "environment": "ai-research-env", "version": "1.0.0",
            "tasks": list(TASKS.keys()), "actions": VALID_ACTIONS}


@app.get("/openenv.yaml", response_class=PlainTextResponse)
def serve_openenv_yaml() -> str:
    """Serve the openenv.yaml spec — required by `openenv validate`."""
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "openenv.yaml")
    yaml_path = os.path.abspath(yaml_path)
    try:
        with open(yaml_path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="openenv.yaml not found")


@app.get("/metadata")
def metadata() -> Dict[str, Any]:
    """Return environment metadata — required by `openenv validate`."""
    return {
        "name": "ai-research-env",
        "version": "1.0.0",
        "description": (
            "AI-Research-Env is a simulation platform for end-to-end scientific discovery "
            "using AI agents. Agents navigate a structured research workflow — reading papers, "
            "forming hypotheses, designing/running experiments, analysing results, and producing "
            "final conclusions — across three ML research domains."
        ),
        "license": "MIT",
        "tags": ["openenv", "ai-research", "scientific-discovery", "rl-environment"],
        "tasks": list(TASKS.keys()),
    }


@app.get("/schema")
def schema() -> Dict[str, Any]:
    """Return action/observation/state schemas — required by `openenv validate`."""
    return {
        "action": {
            "type": "object",
            "description": "Research action with type and content.",
            "properties": {
                "action_type": {"type": "string", "enum": VALID_ACTIONS},
                "content": {"type": "string", "description": "Agent's free-text input"},
                "parameters": {"type": "object", "description": "Optional structured params"},
            },
        },
        "observation": {
            "type": "object",
            "description": "Research observation returned at each step.",
            "properties": {
                "task_id": {"type": "string"},
                "task_name": {"type": "string"},
                "difficulty": {"type": "string"},
                "research_context": {"type": "string"},
                "current_phase": {"type": "string"},
                "allowed_actions": {"type": "array", "items": {"type": "string"}},
                "last_feedback": {"type": "string"},
                "step_number": {"type": "integer"},
                "max_steps": {"type": "integer"},
                "progress": {"type": "object"},
                "hints": {"type": "array", "items": {"type": "string"}},
            },
        },
        "state": {
            "type": "object",
            "description": "Episode metadata returned by state().",
            "properties": {
                "task_id": {"type": "string"},
                "task_name": {"type": "string"},
                "step_number": {"type": "integer"},
                "cumulative_reward": {"type": "number"},
                "done": {"type": "boolean"},
                "phases_completed": {"type": "array", "items": {"type": "string"}},
                "history": {"type": "array", "items": {"type": "object"}},
            },
        },
    }


@app.post("/mcp")
async def mcp_endpoint(body: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal JSON-RPC 2.0 MCP endpoint — required by `openenv validate`."""
    jsonrpc = body.get("jsonrpc", "2.0")
    req_id = body.get("id", 1)
    method = body.get("method", "")

    if method == "initialize":
        return {
            "jsonrpc": jsonrpc, "id": req_id,
            "result": {
                "name": "ai-research-env", "version": "1.0.0",
                "description": "AI-Research-Env — Scientific Discovery Environment",
                "capabilities": ["reset", "step", "state"],
            },
        }
    if method == "describe":
        return {
            "jsonrpc": jsonrpc, "id": req_id,
            "result": {"environment": "ai-research-env", "tasks": list(TASKS.keys())},
        }
    return {"jsonrpc": "2.0", "id": req_id,
            "result": {"environment": "ai-research-env", "method": method, "status": "ok"}}


# ------------------------------------------------------------------
# OpenEnv core: reset / step / state
# ------------------------------------------------------------------

@app.post("/reset")
def reset() -> Dict[str, Any]:
    """Start a new episode. Returns initial observation. No body required."""
    sid = uuid.uuid4().hex[:12]
    env = ResearchEnv(task_name="cv-classification")
    obs = env.reset()
    _sessions[sid] = env
    return {
        "session_id": sid,
        "observation": obs.model_dump(),
    }


@app.post("/step")
def step(body: Dict[str, Any]) -> Dict[str, Any]:
    """Submit an action and advance the episode by one step."""
    session_id = body.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    env = _get(session_id)
    action_data = body.get("action", body)
    action = Action(
        action_type=str(action_data.get("action_type", "read_paper")),
        content=str(action_data.get("content", "")),
        parameters=action_data.get("parameters"),
    )
    try:
        result: StepResult = env.step(action)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(400, str(e))
    return {
        "session_id": session_id,
        "observation": result.observation.model_dump(),
        "reward": float(result.reward),
        "done": bool(result.done),
        "info": result.info,
    }


@app.get("/state")
def state(session_id: str) -> Dict[str, Any]:
    """Return current episode metadata without advancing the episode."""
    return {"session_id": session_id, "state": _get(session_id).state().model_dump()}


@app.post("/close")
def close_session(body: Dict[str, Any]) -> Dict[str, Any]:
    """Release server-side session resources."""
    session_id = body.get("session_id")
    if session_id and session_id in _sessions:
        del _sessions[session_id]
    return {"status": "closed", "session_id": session_id}


@app.get("/tasks")
def list_tasks():
    return {
        "environment": "ai-research-env",
        "version": "1.0.0",
        "tasks": {name: {"difficulty": cfg["difficulty"], "max_steps": cfg["max_steps"],
                         "phases": cfg["phases"]} for name, cfg in TASKS.items()},
    }


# Serve built React frontend (after `npm run build`)
_static = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static):
    app.mount("/assets", StaticFiles(directory=os.path.join(_static, "assets")), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        # API routes already handled above; everything else → SPA
        f = os.path.join(_static, full_path)
        if os.path.isfile(f):
            return FileResponse(f)
        idx = os.path.join(_static, "index.html")
        if os.path.isfile(idx):
            return FileResponse(idx)
        return {"status": "ok"}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
