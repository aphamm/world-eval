"""
Convert a LeRobot parquet/video dataset into ACT-style HDF5 files using Modal.
"""

import modal

from config import mount_path, num_cpus, timeout, train_image, vol

app = modal.App("lora-finetune")


@app.function(
    image=train_image,
    volumes={mount_path: vol},
    cpu=num_cpus,
    gpu="A100-80GB:4",
    timeout=timeout,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def extract_latents():
    import subprocess
    import sys

    from config import mount_path

    subprocess.run(
        [
            "python3",
            "train_wan_t2v_act_embed.py",
            "--task",
            "train",
            "--train_architecture",
            "lora",
            "--dataset_path",
            f"{str(mount_path)}/data/act_dataset/",
            "--output_path",
            f"{str(mount_path)}/weights",
            "--dit_path",
            ",".join(
                [
                    f"{str(mount_path)}/models/diffusion_pytorch_model-00001-of-00007.safetensors",
                    f"{str(mount_path)}/models/diffusion_pytorch_model-00002-of-00007.safetensors",
                    f"{str(mount_path)}/models/diffusion_pytorch_model-00003-of-00007.safetensors",
                    f"{str(mount_path)}/models/diffusion_pytorch_model-00004-of-00007.safetensors",
                    f"{str(mount_path)}/models/diffusion_pytorch_model-00005-of-00007.safetensors",
                    f"{str(mount_path)}/models/diffusion_pytorch_model-00006-of-00007.safetensors",
                    f"{str(mount_path)}/models/diffusion_pytorch_model-00007-of-00007.safetensors",
                ]
            ),
            "--image_encoder_path",
            f"{str(mount_path)}/models/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            "--steps_per_epoch",
            "60",  # if effective_batch_size = 4 with 4 GPUs, then 250 training examples / 4 ≈ 62 steps/epoch
            "--max_epochs",
            "40",
            "--dataloader_num_workers",
            "8",
            "--learning_rate",
            "1e-5",  # original 1e-4 is too high for 17× less data
            "--training_strategy",
            "deepspeed_stage_2",
            "--lora_rank",
            "16",
            "--lora_alpha",
            "16",
            "--lora_target_modules",
            "q,k,v,o,ffn.0,ffn.2,action_alpha,action_proj.0,action_proj.2",
            "--accumulate_grad_batches",  # effective_batch_size = batch_size * accumulate_grad_batches * num_gpus
            "1",
            "--use_gradient_checkpointing",  # reduces memory by recomputing activations during backward pass
            "--action_alpha",
            "0.3",
            "--action_dim",
            "384",
            "--encode_mode",
            "act",
            "--action_encoded_path",
            f"{str(mount_path)}/data/act_dataset/train/all_actions.pt",
        ],
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True,
    )


@app.local_entrypoint()
def main():
    extract_latents.remote()
