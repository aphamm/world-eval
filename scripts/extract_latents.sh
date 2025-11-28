
#! /bin/bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python train_wan_t2v_act_embed.py \
--task data_process \
--dataset_path data/act_dataset \
--output_path ./models \
--text_encoder_path "Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth" \
--image_encoder_path "Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
--vae_path "Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth" \
--tiled \
--num_frames 81 \
--height 480 \
--width 480 \
--encode_mode act \
--samples_per_file 5 \
--action_encoded_path wanvideo/data/act_dataset/train/all_actions.pt
