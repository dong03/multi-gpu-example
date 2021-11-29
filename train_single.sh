GPU=$1
cd train
CUDA_VISIBLE_DEVICES=$GPU python train.py --fp16 True
