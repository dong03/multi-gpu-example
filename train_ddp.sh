GPU=$1
GPU_num=$2 #节点数，单机多卡则为卡数
FP16=$3
cd train
CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.run --nproc_per_node=$GPU_num \
train_ddp.py --fp16 $FP16 \
--resume /home/dcb/code/mutli-GPU-example/train/ckpt/model_1.pt
