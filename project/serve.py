import importlib
import os
import sys

import uvicorn

d = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(d, 'vendor'))
sys.path.insert(0, d)
os.chdir(d)
app = importlib.import_module("deltai_app").app
port = int(os.environ.get("PORT", 8000))
print(f"Starting on port {port} from {d}", flush=True)
uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
