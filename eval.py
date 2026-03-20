"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
import os

# Avoid OpenMP SHM permission errors by preloading system libgomp if available.
if os.environ.get("DP_LAUNCHED_WITH_SYSTEM_GOMP") != "1":
    for libgomp_path in (
        "/usr/lib/x86_64-linux-gnu/libgomp.so.1",
        "/lib/x86_64-linux-gnu/libgomp.so.1",
    ):
        if os.path.exists(libgomp_path):
            preload = os.environ.get("LD_PRELOAD", "")
            if libgomp_path not in preload.split(":"):
                os.environ["DP_LAUNCHED_WITH_SYSTEM_GOMP"] = "1"
                os.environ["LD_PRELOAD"] = (
                    f"{libgomp_path}:{preload}" if preload else libgomp_path
                )
                os.execv(sys.executable, [sys.executable] + sys.argv)
            break

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

# Work around OpenMP shared memory permission errors in restricted environments.
os.environ.setdefault("KMP_SHM_DISABLE", "1")
# Work around protobuf incompatibility with newer generated descriptors.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
import huggingface_hub as hf_hub

if not hasattr(hf_hub, "cached_download") and hasattr(hf_hub, "hf_hub_download"):
    hf_hub.cached_download = hf_hub.hf_hub_download
from diffusion_policy.workspace.base_workspace import BaseWorkspace

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint, output_dir, device):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.", flush=True)
        device = "cpu"
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
