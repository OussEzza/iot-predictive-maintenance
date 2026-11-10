"""
Launcher script to run both FastAPI backend and Streamlit dashboard concurrently.
Usage:
    python run_dashboard.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent

def main():
    print("=" * 70)
    print("Launching IoT Sensor Predictive Maintenance System")
    print("=" * 70)
    print("1. Starting FastAPI Model-Serving Backend on http://127.0.0.1:8000 ...")
    
    api_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app_api:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(DEMO_DIR),
    )

    # Wait 2 seconds for backend to initialize
    time.sleep(2)

    print("2. Starting Streamlit Frontend Dashboard on http://localhost:8501 ...")
    try:
        streamlit_process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "app_ui.py", "--server.port", "8501"],
            stdout=sys.stdout,
            stderr=sys.stderr,
            cwd=str(DEMO_DIR),
        )
        streamlit_process.wait()
    except KeyboardInterrupt:
        print("\nStopping services...")
    finally:
        api_process.terminate()
        print("Services stopped.")

if __name__ == "__main__":
    main()
