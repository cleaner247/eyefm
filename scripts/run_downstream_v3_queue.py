from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


TASKS = [
    "pd_related_5class",
    "pd_binary",
    "epilepsy_binary",
    "detox_binary",
    "migraine_binary",
    "ad_binary",
    "mci_original_only_binary",
    "mci_matched_binary_random_seed20260621",
]
MODES = ["scratch", "linear_probe", "partial", "full"]


def read_config_list(path: Path) -> list[str]:
    configs: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        configs.append(line)
    return configs


def build_jobs(configs: list[str] | None = None, *, config_list_file: str | None = None) -> list[Path]:
    items: list[str] = []
    if config_list_file:
        items.extend(read_config_list(Path(config_list_file)))
    if configs:
        items.extend(configs)
    if items:
        return [Path(item) for item in items]
    jobs: list[Path] = []
    for task in TASKS:
        for mode in MODES:
            jobs.append(Path("configs/downstream") / f"{task}_{mode}.yaml")
    return jobs


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def public_job_info(info: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in info.items() if key not in {"handle", "proc"}}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def config_output_dir(config_path: Path) -> Path | None:
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        return None
    output_dir = cfg.get("experiment", {}).get("output_dir")
    if not output_dir:
        return None
    return Path(str(output_dir))


def config_completed(config_path: Path) -> bool:
    output_dir = config_output_dir(config_path)
    return output_dir is not None and (output_dir / "metrics_final.json").exists()


def config_resume_checkpoint(config_path: Path) -> Path | None:
    output_dir = config_output_dir(config_path)
    if output_dir is None or (output_dir / "metrics_final.json").exists():
        return None
    checkpoint = output_dir / "checkpoint_last.pt"
    return checkpoint if checkpoint.exists() else None


def pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def parse_attach_spec(spec: str) -> dict[str, object]:
    parts = spec.split(":", 3)
    if len(parts) != 4:
        raise ValueError("--attach must use gpu:pid:config:log")
    gpu, pid, config, log = parts
    return {
        "gpu": gpu,
        "pid": int(pid),
        "config": config,
        "log": log,
        "start_time": time.time(),
        "attached": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="1,2,3,4")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Optional explicit config list to run instead of the default 7-task downstream matrix.",
    )
    parser.add_argument(
        "--config-list-file",
        default=None,
        help="Optional newline-delimited config list. Blank lines and # comments are ignored.",
    )
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--resume-status", default=None)
    parser.add_argument(
        "--attach",
        action="append",
        default=[],
        help="Attach an already-running job as gpu:pid:config:log",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise SystemExit("At least one GPU id is required")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir or f"outputs/downstream_v3_logs/run_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args.configs, config_list_file=args.config_list_file)
    missing = [str(path) for path in jobs if not path.exists()]
    if missing:
        raise SystemExit(f"Missing downstream configs: {missing}")

    if args.dry_run:
        for idx, cfg in enumerate(jobs):
            print(f"GPU {gpus[idx % len(gpus)]}: {cfg}")
        return 0

    completed: list[dict[str, object]] = []
    if args.resume_status:
        status_path = Path(args.resume_status)
        if status_path.exists():
            previous_status = read_json(status_path)
            completed = [
                item
                for item in previous_status.get("completed", [])
                if int(item.get("returncode", 0)) == 0
            ]
    completed_configs = {str(item["config"]) for item in completed}
    for cfg in jobs:
        if str(cfg) not in completed_configs and config_completed(cfg):
            completed.append(
                {
                    "config": str(cfg),
                    "gpu": None,
                    "log": str(log_dir / f"{cfg.stem}.log"),
                    "pid": None,
                    "start_time": None,
                    "returncode": 0,
                    "duration_sec": None,
                    "source": "metrics_final",
                }
            )
            completed_configs.add(str(cfg))

    attached_jobs = [parse_attach_spec(spec) for spec in args.attach]
    attached_configs = {str(item["config"]) for item in attached_jobs}
    pending = [cfg for cfg in jobs if str(cfg) not in completed_configs and str(cfg) not in attached_configs]
    running: dict[int, dict[str, object]] = {}
    free_gpus = list(gpus)
    for item in attached_jobs:
        gpu = str(item["gpu"])
        if gpu in free_gpus:
            free_gpus.remove(gpu)
        running[int(item["pid"])] = item

    def write_status(*, failures: list[dict[str, object]] | None = None) -> None:
        payload: dict[str, object] = {
            "completed": completed,
            "running": [public_job_info(item) for item in running.values()],
            "pending": [str(p) for p in pending],
        }
        if failures is not None:
            payload["failures"] = failures
        write_json(log_dir / "status.json", payload)

    def launch(cfg: Path, gpu: str) -> None:
        name = cfg.stem
        log_path = log_dir / f"{name}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTHONUNBUFFERED"] = "1"
        command = [sys.executable, "-m", "eyemae.finetune", "--config", str(cfg)]
        resume_checkpoint = config_resume_checkpoint(cfg)
        if resume_checkpoint is not None:
            command.extend(["--resume", str(resume_checkpoint)])
        log_existed = log_path.exists()
        handle = log_path.open("ab" if log_existed else "wb")
        if log_existed:
            handle.write(
                f"\n--- queue launch {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"config={cfg} gpu={gpu} ---\n".encode("utf-8")
            )
            handle.flush()
        proc = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
        running[proc.pid] = {
            "config": str(cfg),
            "gpu": gpu,
            "log": str(log_path),
            "pid": proc.pid,
            "start_time": time.time(),
            "resume": str(resume_checkpoint) if resume_checkpoint is not None else None,
            "proc": proc,
            "handle": handle,
        }
        resume_note = f" resume={resume_checkpoint}" if resume_checkpoint is not None else ""
        print(f"LAUNCH gpu={gpu} pid={proc.pid} config={cfg} log={log_path}{resume_note}", flush=True)
        write_status()

    write_status()
    while pending or running:
        while pending and free_gpus:
            launch(pending.pop(0), free_gpus.pop(0))

        time.sleep(float(args.poll_seconds))

        for key, info in list(running.items()):
            proc = info.get("proc")
            if isinstance(proc, subprocess.Popen):
                ret = proc.poll()
                if ret is None:
                    continue
            else:
                pid = int(info["pid"])
                if pid_alive(pid):
                    continue
                ret = 0 if config_completed(Path(str(info["config"]))) else 1
            handle = info.pop("handle", None)
            if handle is not None:
                handle.close()
            duration = time.time() - float(info["start_time"])
            result = {
                **public_job_info(info),
                "returncode": int(ret),
                "duration_sec": duration,
            }
            completed.append(result)
            free_gpus.append(str(info["gpu"]))
            del running[key]
            print(
                f"DONE gpu={result['gpu']} pid={result['pid']} rc={ret} "
                f"duration_sec={duration:.1f} config={result['config']}",
                flush=True,
            )
            write_status()
        write_status()

    failures = [item for item in completed if int(item["returncode"]) != 0]
    write_status(failures=failures)
    print(f"Finished {len(completed)} downstream jobs; failures={len(failures)}; log_dir={log_dir}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
