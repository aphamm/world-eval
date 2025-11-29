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
            "--encode_mode",
            "act",
            "--samples_per_file",
            "5",
            "--action_encoded_path",
            f"{str(mount_path)}/data/act_dataset/train/all_actions.pt",
            "--dataloader_num_workers",
            "8",
        ],
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True,
    )


@app.local_entrypoint()
def main():
    extract_latents.remote()
