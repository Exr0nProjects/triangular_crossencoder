VISIBLE_GPU=${1:-1}
WANDB_PROJECT=sentence_transformers_wes WANDB_ENTITY=llo-qaval CUDA_VISIBLE_DEVICES=$VISIBLE_GPU python3 main.py

