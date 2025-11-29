"""
Convert a LeRobot parquet/video dataset into ACT-style HDF5 files using Modal.
"""

import pathlib

import modal

app = modal.App("extract-latents")
vol = modal.Volume.from_name("my-volume", create_if_missing=True, version=2)
mount_path = pathlib.Path("/mnt")


image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu22.04", add_python="3.10")
    .run_commands(
        "apt-get update && apt-get install -y --no-install-recommends \
            build-essential cmake ninja-build pkg-config ffmpeg bash \
            && python3 -m pip install --upgrade --no-cache-dir pip wheel setuptools packaging \
            && rm -rf /var/lib/apt/lists/*"
    )
    .uv_pip_install(
        "torch>=2.0.0",
        "torchvision",
        "cupy-cuda12x",
        "transformers==4.46.2",
        "controlnet-aux==0.0.7",
        "imageio",
        "imageio[ffmpeg]",
        "safetensors",
        "einops",
        "sentencepiece",
        "protobuf",
        "modelscope",
        "ftfy",
        "peft==0.13.0",
        "lightning",
        "pandas",
        "h5py",
        "pytest",
        "litmodels",
        "lightning[extra]",
    )
    .add_local_dir("diffsynth", remote_path="/root/diffsynth", copy=True)
    .add_local_file(
        "train_wan_t2v_act_embed.py", remote_path="/root/train_wan_t2v_act_embed.py"
    )
)


@app.function(
    image=image,
    volumes={mount_path: vol},
    timeout=60 * 60 * 8,
    cpu=4.0,
    gpu="A100-80GB",
)
def extract_latents():
    import subprocess
    import sys

    subprocess.run(["ls", "-l"], check=True)

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
            f"{str(mount_path)}/models",
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
            "500",
            "--max_epochs",
            "40",
            "--learning_rate",
            "1e-4",
            "--lora_rank",
            "16",
            "--lora_alpha",
            "16",
            "--lora_target_modules",
            "q,k,v,o,ffn.0,ffn.2,action_alpha,action_proj.0,action_proj.2",
            "--accumulate_grad_batches",
            "1",
            "--use_gradient_checkpointing",
            "--action_alpha",
            "0.3",
            "--action_dim",
            "384",
            "--encode_mode",
            "act",
            "--action_encoded_path",
            f"{str(mount_path)}/data/act_dataset/train/all_actions.pt",
            "--version",
            "lora_act_alpha_0.3_act",
        ],
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True,
    )


@app.local_entrypoint()
def main():
    extract_latents.remote()
