# Multi-GPU-Example for AIMC-Lab
## Discribe
+ Multi gpus: DDP(DistributedDataParallel)
+ Half precision: Apex
+ Corresponding single gpu version (for comparision)

## Requirement
+ Install [nvidia-apex](https://github.com/NVIDIA/apex) and move it to current directory.
+ torch >= 1.9.1
## Usage
+ multi gpus
```angular2
bash train_ddp.sh GPU_ID GPU_NUM
```

+ single gpu
```angular2
bash train_single.sh GPU_ID
```
