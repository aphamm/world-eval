#!/bin/bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python infer.py \
--lora_path "" \
--meta_path "data/act_dataset/metadata.json" \
--output_subdir "lora_act_alpha_0.3_act_ep30" \
--action \
--action_alpha 0.3 \
--action_dim   384 \
--action_encoded_path "data/act_dataset/train/all_actions.pt"
