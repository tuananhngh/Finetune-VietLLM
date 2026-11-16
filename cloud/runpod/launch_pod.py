# /// script
# requires-python = ">=3.11"
# dependencies = ["runpod>=2.2"]
# ///
"""Create a RunPod GPU pod that trains a config and (optionally) stops itself when done.

    export RUNPOD_API_KEY=...  HF_TOKEN=...  WANDB_API_KEY=...
    uv run cloud/runpod/launch_pod.py --list-gpus        # uv installs `runpod` in a throwaway env
    uv run cloud/runpod/launch_pod.py \
        --image ghcr.io/<you>/vietllm:latest \
        --gpu "NVIDIA GeForce RTX 4090" \
        --config configs/sft/qwen3-8b-qlora-bactrian.yaml --auto-stop

Use --network-volume-id to keep checkpoints / HF cache across pods (the volume is mounted at /workspace and
training resumes from the last checkpoint automatically).
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    try:
        import runpod
    except ImportError:
        sys.exit("Run this script with `uv run cloud/runpod/launch_pod.py` (installs runpod automatically)")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list-gpus", action="store_true", help="Print available GPU type ids and exit")
    p.add_argument("--name", default="vietllm-train")
    p.add_argument(
        "--image", default=os.environ.get("VIETLLM_IMAGE"), help="Docker image built from docker/Dockerfile"
    )
    p.add_argument("--gpu", default="NVIDIA GeForce RTX 4090", help="RunPod gpu_type_id")
    p.add_argument("--gpu-count", type=int, default=1)
    p.add_argument("--cloud-type", choices=["SECURE", "COMMUNITY", "ALL"], default="SECURE")
    p.add_argument("--config", default=None, help="Run config; omit for an idle pod you SSH into")
    p.add_argument("--accel-config", default=None)
    p.add_argument(
        "--torch-extra",
        default=None,
        choices=["cu126", "cu132"],
        help="Only for stock images: CUDA build of torch to install (default cu126)",
    )
    p.add_argument("--extra-args", default="", help='CLI overrides, e.g. "--max_steps 100"')
    p.add_argument(
        "--volume-gb", type=int, default=150, help="Pod volume at /workspace (ignored with network volume)"
    )
    p.add_argument("--container-disk-gb", type=int, default=50)
    p.add_argument("--network-volume-id", default=None)
    p.add_argument("--data-center-id", default=None, help="Must match the network volume's data center")
    p.add_argument("--auto-stop", action="store_true", help="Stop the pod when training ends")
    args = p.parse_args()

    runpod.api_key = os.environ.get("RUNPOD_API_KEY") or sys.exit("RUNPOD_API_KEY is not set")

    if args.list_gpus:
        for gpu in sorted(runpod.get_gpus(), key=lambda g: g.get("memoryInGb", 0)):
            print(f"{gpu['id']:<40} {gpu.get('memoryInGb', '?')} GB")
        return
    if not args.image:
        sys.exit("--image is required (build and push docker/Dockerfile first)")

    env = {
        "WORKSPACE": "/workspace",
        "AUTO_STOP": "true" if args.auto_stop else "false",
    }
    for key in ("HF_TOKEN", "WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    if args.config:
        env["CONFIG"] = args.config
    if args.accel_config:
        env["ACCEL_CONFIG"] = args.accel_config
    if args.torch_extra:
        env["TORCH_EXTRA"] = args.torch_extra
    if args.extra_args:
        env["EXTRA_ARGS"] = args.extra_args

    pod = runpod.create_pod(
        name=args.name,
        image_name=args.image,
        gpu_type_id=args.gpu,
        gpu_count=args.gpu_count,
        cloud_type=args.cloud_type,
        support_public_ip=True,
        start_ssh=True,
        volume_in_gb=0 if args.network_volume_id else args.volume_gb,
        container_disk_in_gb=args.container_disk_gb,
        volume_mount_path="/workspace",
        network_volume_id=args.network_volume_id,
        data_center_id=args.data_center_id,
        ports="22/tcp,8000/http,6006/http",
        env=env,
    )
    pod_id = pod["id"]
    print(f"Created pod {pod_id}")
    print(f"  console : https://www.runpod.io/console/pods/{pod_id}")
    print(
        f"  logs    : runpodctl logs {pod_id}  (or the web terminal: tail -f /workspace/outputs/train-*.log)"
    )
    print(f"  stop    : runpodctl stop pod {pod_id}   |   remove: runpodctl remove pod {pod_id}")


if __name__ == "__main__":
    main()
