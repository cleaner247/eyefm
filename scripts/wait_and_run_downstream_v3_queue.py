from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Status is not a JSON object: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-json", required=True)
    parser.add_argument("--config-list-file", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--gpus", default="1,2,3,4")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--queue-poll-seconds", type=float, default=30.0)
    parser.add_argument(
        "--allow-upstream-failures",
        action="store_true",
        help=(
            "Launch the resume queue even if the upstream queue reports failures. "
            "Use this when failed jobs have checkpoint_last.pt and should be resumed."
        ),
    )
    args = parser.parse_args()

    status_path = Path(args.status_json)
    while True:
        if not status_path.exists():
            print(f"WAIT status_json_missing={status_path}", flush=True)
            time.sleep(float(args.poll_seconds))
            continue
        status = read_json(status_path)
        completed = len(status.get("completed", []))
        running = len(status.get("running", []))
        pending = len(status.get("pending", []))
        failures = len(status.get("failures", []))
        print(
            f"WAIT completed={completed} running={running} pending={pending} failures={failures}",
            flush=True,
        )
        if failures and not args.allow_upstream_failures:
            print("ABORT upstream queue reported failures; not launching tail queue", flush=True)
            return 1
        if running == 0 and pending == 0 and completed > 0:
            break
        time.sleep(float(args.poll_seconds))

    command = [
        sys.executable,
        "scripts/run_downstream_v3_queue.py",
        "--gpus",
        args.gpus,
        "--log-dir",
        args.log_dir,
        "--poll-seconds",
        str(float(args.queue_poll_seconds)),
        "--config-list-file",
        args.config_list_file,
        "--resume-status",
        args.status_json,
    ]
    print("LAUNCH_TAIL " + " ".join(command), flush=True)
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
