VISIBLE_GPU=${1:-1}
WANDB_PROJECT=crossencoder_baseline WANDB_ENTITY=llo-qaval CUDA_VISIBLE_DEVICES=$VISIBLE_GPU python3 main.py

