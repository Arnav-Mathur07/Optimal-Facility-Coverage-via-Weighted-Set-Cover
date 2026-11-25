"""
Optimal Facility Coverage Dashboard & Web Simulator Launcher.
Starts both the Streamlit Dashboard (Port 8501) and MotionSites Web Lab (Port 8080) in one command.

Usage:
    python run.py
"""

import sys
import subprocess
import time
import webbrowser
import signal
from pathlib import Path

def main():
    root_dir = Path(__file__).resolve().parent

    print("\n" + "=" * 65)
    print("  DAA PBL: OPTIMAL FACILITY COVERAGE VIA WEIGHTED SET COVER")
    print("=" * 65)
    print("  [1] Starting Streamlit Analytical Dashboard on Port 8501...")
    print("  [2] Starting MotionSites Cinematic Web Lab on Port 8080...")
    print("-" * 65)

    # 1. Start Streamlit
    cmd_streamlit = [
        sys.executable,
        "-m", "streamlit", "run",
        str(root_dir / "app.py"),
        "--server.port", "8501",
        "--server.headless", "true"
    ]
    p_streamlit = subprocess.Popen(cmd_streamlit, cwd=root_dir)

    # 2. Start HTTP server for MotionSites Web Lab
    cmd_web = [
        sys.executable,
        "-m", "http.server", "8080"
    ]
    p_web = subprocess.Popen(cmd_web, cwd=root_dir)

    time.sleep(2)

    print("\n  >>> BOTH SERVERS ARE LIVE! <<<")
    print("  * Streamlit Dashboard : http://localhost:8501")
    print("  * MotionSites Web Lab : http://localhost:8080")
    print("-" * 65)
    print("  Press Ctrl+C to stop both servers.\n")

    # Optionally open browser to Streamlit
    try:
        webbrowser.open("http://localhost:8501")
    except Exception:
        pass

    try:
        while True:
            time.sleep(1)
            # If either process exits unexpectedly, break
            if p_streamlit.poll() is not None:
                print("Streamlit process stopped.")
                break
            if p_web.poll() is not None:
                print("Web server process stopped.")
                break
    except KeyboardInterrupt:
        print("\nStopping all servers...")
    finally:
        p_streamlit.terminate()
        p_web.terminate()
        p_streamlit.wait()
        p_web.wait()
        print("Servers stopped cleanly.")

if __name__ == "__main__":
    main()
