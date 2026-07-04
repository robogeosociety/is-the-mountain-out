import time
import random
import subprocess
from datetime import datetime, UTC


def run_capture(session_id: str):
    print(f"[{datetime.now(UTC).isoformat()}] Running capture...")
    # Using 'uv run collect once' to leverage existing logic for a single capture
    cmd = [
        "uv",
        "run",
        "collect",
        "once",
        "--config",
        "mountain.toml",
        "--data-root",
        "data/out_captures",
        "--session-id",
        session_id,
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    session_id = f"out_{int(time.time())}"
    print(f"Starting 'Out' capture job (Session: {session_id})...")

    # 10 captures over 3600 seconds (1 hour)
    total_duration = 3600
    num_captures = 10

    # Generate 10 random delay points in the 1 hour window
    delays = sorted([random.randint(0, total_duration) for _ in range(num_captures)])

    current_time = 0
    for i, delay in enumerate(delays):
        wait_time = delay - current_time
        if wait_time > 0:
            print(f"Waiting {wait_time}s for capture {i + 1}/10...")
            time.sleep(wait_time)

        try:
            run_capture(session_id)
        except Exception as e:
            print(f"Capture failed: {e}")

        current_time = delay

    print("All captures completed.")
