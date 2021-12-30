CUDA_DEVICE=${1:-1}
WANDB_PROJECT=tri-roberta-baseline WANDB_ENTITY=llo-qaval CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 main.py

