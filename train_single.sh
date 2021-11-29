GPU=$1
FP16=True
cd train
CUDA_VISIBLE_DEVICES=$GPU python train.py --fp16 $FP16
