"""
Convert a LeRobot parquet/video dataset into ACT-style HDF5 files using Modal.
"""

import modal

from config import mount_path, num_cpus, timeout, vol

app = modal.App("generate-train-data")


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
        "lightning[extra]",
    )
    .add_local_dir("diffsynth", remote_path="/root/diffsynth", copy=True)
    .add_local_file(
        "train_wan_t2v_act_embed.py", remote_path="/root/train_wan_t2v_act_embed.py"
    )
    .add_local_file("config.py", remote_path="/root/config.py")
)


@app.function(
    image=image,
    volumes={mount_path: vol},
    cpu=num_cpus,
    gpu="L40S",
    timeout=timeout,
)
def extract_latents():
    import subprocess
    import sys

    subprocess.run(
        [
            "python3",
            "train_wan_t2v_act_embed.py",
            "--task",
            "data_process",
            "--dataset_path",
            f"{str(mount_path)}/data/act_dataset",
            "--output_path",
            f"{str(mount_path)}/models",
            "--text_encoder_path",
            f"{str(mount_path)}/models/models_t5_umt5-xxl-enc-bf16.pth",
            "--image_encoder_path",
            f"{str(mount_path)}/models/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            "--vae_path",
            f"{str(mount_path)}/models/Wan2.1_VAE.pth",
            "--tiled",
            "--num_frames",
            "81",
            "--height",
            "480",
            "--width",
            "480",
            "--dataloader_num_workers",
            "8",
            "--encode_mode",
            "act",
            "--samples_per_file",
            "5",
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
