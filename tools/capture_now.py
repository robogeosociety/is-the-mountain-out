#!/usr/bin/env python3
import subprocess
import sys
import argparse
import uuid
from pathlib import Path


def trigger_capture(session_id=None):
    """Triggers a single capture by running a unique Nomad batch job."""
    hcl_path = Path("nomad/once.hcl")
    if not hcl_path.exists():
        # Check relative to script location
        hcl_path = Path(__file__).parent.parent / "nomad" / "once.hcl"

    if not hcl_path.exists():
        print(f"Error: {hcl_path} not found.")
        sys.exit(1)

    unique_id = str(uuid.uuid4())[:8]
    job_name = f"adhoc-{unique_id}"

    # Read HCL and replace job name
    hcl_content = hcl_path.read_text()
    hcl_content = hcl_content.replace(
        'job "mountain-capture-single"', f'job "{job_name}"'
    )

    cmd = ["nomad", "job", "run"]
    if session_id:
        cmd.extend(["-var", f"session_id={session_id}"])
    cmd.append("-")  # Read from stdin

    print(f"Submitting Nomad job {job_name}...")
    try:
        # Run nomad job run and capture output
        result = subprocess.run(
            cmd, input=hcl_content, capture_output=True, text=True, check=True
        )
        print(f"Nomad job {job_name} submitted successfully.")

        # Extract the Eval ID from output
        for line in result.stdout.splitlines():
            if "Monitoring evaluation" in line:
                eval_id = line.split()[-1].strip('"')
                print(f"Evaluation ID: {eval_id}")
                break

        print("Capture in progress via Nomad (batch job)...")
    except subprocess.CalledProcessError as e:
        print(f"Error submitting Nomad job:\n{e.stderr}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", help="Session ID to use for this capture")
    args = parser.parse_args()
    trigger_capture(args.session_id)
