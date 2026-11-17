#!/usr/bin/env bash
# Train QwenGR00T (Qwen3.5-0.8B frozen + DiT-B) on the LGES sim datasets. Needs the GPU, so run it outside the
# Claude sandbox. Extra arguments override the yaml:
#
#   examples/LGES/train_files/run_lges_train.sh                                  # the full run
#   examples/LGES/train_files/run_lges_train.sh --run_id smoke \
#       --trainer.max_train_steps 50 --trainer.eval_interval 25 --trainer.save_interval 100000
#                                                                                # a short test: speed and VRAM
#
# Measured 2026-11-16 on the RTX 5080 with the GPU otherwise idle: batch 16 takes 13.7 GB for this process (the
# desktop holds another 1.8 GB of the 16.3 GB) and 0.64 s per step. It does not fit next to other GPU jobs;
#   --datasets.vla_data.per_device_batch_size 4 --framework.action_model.repeated_diffusion_steps 4
# takes 6.6 GB. The process stays about 80 s after the last step before it exits.
set -euo pipefail
cd "$(dirname "$0")/../../.."                    # the starVLA root: the yaml's paths start here

export PATH="$PWD/.venv/bin:$PATH"

# This machine has no C compiler and no CUDA toolkit. With a GPU visible, DeepSpeed's import needs both, only to
# probe optional compiled ops that ZeRO-2 training does not use. The two stand-ins in shims/ get the import past
# that (each file says what it does); with gcc and a CUDA toolkit installed, drop these lines and the folder.
SHIMS="$PWD/examples/LGES/train_files/shims"
export PYTHONPATH="$SHIMS/pythonpath${PYTHONPATH:+:$PYTHONPATH}"     # hides triton from DeepSpeed
if ! command -v nvcc >/dev/null && [ -z "${CUDA_HOME:-}" ] && [ ! -d /usr/local/cuda ]; then
  export CUDA_HOME="$SHIMS/cuda_home"                                # answers DeepSpeed's `nvcc -V`
  export TORCH_CUDA_VERSION="$(python -c 'import torch; print(torch.version.cuda)')"
fi
export WANDB_MODE="${WANDB_MODE:-offline}"       # the logs stay on this machine (<run dir>/wandb)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # the weights are local
export NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false

exec accelerate launch \
  --config_file examples/LGES/train_files/deepspeed/accelerate_zero2_1gpu.yaml \
  --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/LGES/train_files/starvla_lges_qwen35_gr00t.yaml \
  "$@"
