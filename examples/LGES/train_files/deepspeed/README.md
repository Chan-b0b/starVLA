DeepSpeed settings for one 16 GB GPU.

`ds_config_1gpu.json` is upstream's `starVLA/config/deepseeds/ds_config.yaml` with three values changed:

| key | upstream | here | why |
|---|---|---|---|
| `reduce_bucket_size` | 5e8 | 5e7 | ZeRO-2 allocates this many bf16 elements per gradient buffer on the GPU: 954 MiB each at 5e8, 95 MiB at 5e7. The largest trainable tensor here has a few million elements. |
| `overlap_comm` | true | false | With `true` ZeRO-2 allocates two of those buffers. There is no communication to overlap on one GPU. |
| `allgather_bucket_size` | 5e8 | 5e7 | Kept equal to the reduce bucket. |

Gradient clipping (1.0) is done by DeepSpeed from this file. `trainer.gradient_clipping` in the training yaml only
switches a call that accelerate turns into a no-op under DeepSpeed.
