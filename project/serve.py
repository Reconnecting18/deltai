import sys, os
d = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(d, 'vendor'))
sys.path.insert(0, d)
os.chdir(d)
from e3n_app import app
port = int(os.environ.get("PORT", 8000))
print(f"Starting on port {port} from {d}", flush=True)
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
