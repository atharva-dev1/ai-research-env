"""Quick server test — verify /reset works with empty body."""
import uvicorn, threading, time, requests
from backend.server import app

server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=7861, log_level="error"))
thread = threading.Thread(target=server.run, daemon=True)
thread.start()
time.sleep(2)

# Test /health
r = requests.get("http://127.0.0.1:7861/health")
status = r.json()["status"]
print(f"GET /health: {r.status_code} -> {status}")
assert r.status_code == 200

# Test /reset with EMPTY body (this is what the evaluator does!)
r = requests.post("http://127.0.0.1:7861/reset")
sid = r.json().get("session_id", "MISSING")
print(f"POST /reset (empty): {r.status_code} -> session_id={sid}")
assert r.status_code == 200, f"FAILED! Got {r.status_code}: {r.text}"

# Test /step
r = requests.post("http://127.0.0.1:7861/step", json={
    "session_id": sid,
    "action_type": "read_paper",
    "content": "Test read paper content"
})
reward = r.json().get("reward", -1)
print(f"POST /step: {r.status_code} -> reward={reward}")
assert r.status_code == 200

print("\nALL TESTS PASSED!")
server.should_exit = True
