CUDA_VISIBLE_DEVICES=0 \
TZ=Asia/Shanghai \
OFFLOAD=true \
TASK_TYPE=sft \
MAX_SEQ_LEN=128 \
MODELING_NAME=llama3_classification \
MODEL_NAME_OR_PATH=/model/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa \
TRAIN_DATA_DIR=./data_samples/sft \
CHECKPOINT_DIR=/model/checkpoint \
TRAIN_BSZ_PER_GPU=1 \
SAVE_STEP_INTERVAL=20 \
RESUME_TRAINING=false \
ZERO_MODEL_INPUT=./checkpoint \
RESUME_DATALOADER=false \
DATALOADER_STATE_PATH=./checkpoint \
deepspeed --master_port 29501 train/trainer/sft/train_stage_sft.py
