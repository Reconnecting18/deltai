import sys, os
d = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(d, 'vendor'))
os.chdir(d)
port = int(os.environ.get("PORT", 8000))
print(f"Starting on port {port}", flush=True)
import uvicorn
uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="warning")
