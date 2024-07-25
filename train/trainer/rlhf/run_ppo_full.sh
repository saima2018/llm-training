#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0


CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1  deepspeed ppo_full.py \
   --actor_model_name_or_path facebook/opt-1.3b \
   --critic_model_name_or_path ../rm/output \
   --actor_zero_stage 3 --critic_zero_stage 3 \
   --num_padding_at_beginning 1 --gradient_accumulation_steps 1 \
   --deepspeed --enable_hybrid_engine \
   --actor_gradient_checkpointing \
   --disable_actor_dropout \
   --output_dir ./output &> ./output/training.log
