import uvicorn
import threading
import webbrowser
import time
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def open_browser():
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    print("") 
    print("  ███████╗██████╗ ███╗   ██╗")
    print("  ██╔════╝╚════██╗████╗  ██║")
    print("  █████╗   █████╔╝██╔██╗ ██║")
    print("  ██╔══╝   ╚═══██╗██║╚██╗██║")
    print("  ███████╗██████╔╝██║ ╚████║")
    print("  ╚══════╝╚═════╝ ╚═╝  ╚═══╝")
    print("")
    print("  Local AI Assistant — Online")
    print("  http://127.0.0.1:8000")
    print("  Press Ctrl+C to shut down")
    print("")

    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="warning")
