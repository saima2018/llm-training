#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0


CUDA_VISIBLE_DEVICES=1,2,4,5 deepspeed rm_full.py --model_name_or_path facebook/opt-350m \
   --num_padding_at_beginning 1 --weight_decay 0.1 --disable_dropout \
   --gradient_accumulation_steps 4 --zero_stage 3 \
   --enable_tensorboard \
   --tensorboard_path ./output \
   --deepspeed --output_dir ./output &> ./output/training.log
