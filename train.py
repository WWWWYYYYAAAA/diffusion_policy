"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
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

os.environ.setdefault("KMP_SHM_DISABLE", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_SILENT", "true")

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import huggingface_hub as hf_hub

if not hasattr(hf_hub, "cached_download") and hasattr(hf_hub, "hf_hub_download"):
    hf_hub.cached_download = hf_hub.hf_hub_download

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    if getattr(cfg, "training", None) is not None:
        device = getattr(cfg.training, "device", None)
        if isinstance(device, str) and device.startswith("cuda"):
            import torch
            if not torch.cuda.is_available():
                print("CUDA requested but not available; falling back to CPU.", flush=True)
                cfg.training.device = "cpu"

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
