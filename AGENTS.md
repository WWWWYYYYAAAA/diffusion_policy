# Repository Guidelines

## priority task
fix the problems when run mamba env create -f conda_environment.yaml` or `conda env create -f conda_environment.yaml. please fix dependencies problem in order to make it can run.

## Project Structure & Module Organization
Core code lives under `diffusion_policy/`. Important subpackages include `policy/` for policy classes, `model/` for diffusion and vision backbones, `workspace/` for train/eval workspaces, `env/` and `env_runner/` for simulation and real-robot environments, and `config/` for Hydra YAML configs. Tests live in `tests/` and generally mirror utility or runner modules, for example `tests/test_replay_buffer.py`. Top-level entry points include `train.py`, `ray_train_multirun.py`, `eval_real_robot.py`, and dataset utilities under `diffusion_policy/scripts/`.

## Build, Test, and Development Commands
Create the main environment with `mamba env create -f conda_environment.yaml` or `conda env create -f conda_environment.yaml`. Install the package in editable mode with `pip install -e .` so local module changes are importable.

Run unit tests with `pytest tests/`. Target a single test file during iteration, for example `pytest tests/test_timestamp_accumulator.py`. Start a training run with `python train.py --config-name=train_diffusion_unet_image_workspace`, or pass a downloaded config with `python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml`. For multi-seed runs, use `python ray_train_multirun.py ...` after starting Ray.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, `snake_case` for modules/functions, `PascalCase` for classes, and descriptive config names such as `train_diffusion_unet_hybrid_workspace.yaml`. Keep new files within the established package layout instead of creating new top-level folders. There is no enforced formatter config in-tree; match surrounding code and keep imports and docstrings consistent. `pyrightconfig.json` excludes generated output directories, so avoid placing source under `data/` or `outputs/`.

## Testing Guidelines
Use `pytest` for automated tests. Name new tests `tests/test_<feature>.py` and keep fixture scope narrow. Prefer small deterministic tests for utilities, buffers, interpolation, and environment wrappers before adding long training-path tests. If a change touches training, configs, or runners, include the exact command used to validate it in your PR.

## Commit & Pull Request Guidelines
Recent history uses short, imperative commit subjects such as `fix typo in rotation_transformer.py` and `pinned llvm-openmp version to avoid cpu affinity bug in pytorch`. Keep commits focused and descriptive. Pull requests should summarize the affected task or subsystem, list validation commands, note any dataset or hardware assumptions, and attach metrics, logs, or rollout screenshots/videos for behavior changes.
