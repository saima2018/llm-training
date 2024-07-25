CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
TZ=Asia/Shanghai \
OFFLOAD=false \
TASK_TYPE=sft \
MAX_SEQ_LEN=1000 \
MODELING_NAME=chatglm_v2 \
MODEL_NAME_OR_PATH=/data4/models/huggingface/chatglm/6B_v2 \
TRAIN_DATA_DIR=./data_samples/sft \
CHECKPOINT_DIR=./checkpoint \
TRAIN_BSZ_PER_GPU=1 \
SAVE_STEP_INTERVAL=20 \
RESUME_TRAINING=false \
ZERO_MODEL_INPUT=./checkpoint \
RESUME_DATALOADER=false \
DATALOADER_STATE_PATH=./checkpoint \
deepspeed --master_port 29501 train/trainer/sft/train_stage.py
