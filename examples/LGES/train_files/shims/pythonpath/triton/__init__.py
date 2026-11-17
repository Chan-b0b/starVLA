"""Stand-in that hides triton from the training processes. run_lges_train.sh puts this folder on PYTHONPATH.

DeepSpeed imports its triton inference kernels whenever triton can be imported and the GPU is recent. Training does
not use them, but triton builds a C helper at that import, and this machine has no C compiler, so the launch failed
with "Failed to find C compiler". With triton hidden, DeepSpeed skips those kernels.

Side effect: torch.compile does not work in these processes (starVLA's trainer does not use it). With gcc or clang
installed, this folder and its PYTHONPATH line in run_lges_train.sh can be removed.
"""
raise ImportError("triton is hidden for starVLA training: see examples/LGES/train_files/shims/pythonpath/triton")
