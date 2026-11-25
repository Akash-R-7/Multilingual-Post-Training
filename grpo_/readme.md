To start GRPO training (colocate mode):
```
CUDA_VISIBLE_DEVICES=<GPU_ID> \
    NCCL_NVLS_ENABLE=0 \
    accelerate launch \
    --num_processes <N> \
    multi_train_grpo_indic.py
```
