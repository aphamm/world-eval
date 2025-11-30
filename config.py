import pathlib

import modal

mount_path = pathlib.Path("/mnt")

hf_image = modal.Image.debian_slim(python_version="3.10").uv_pip_install(
    "huggingface-hub==0.35.3"
)

act_image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu22.04", add_python="3.10")
    .run_commands(
        "apt-get update && apt-get install -y --no-install-recommends "
        "build-essential ca-certificates clang ffmpeg libsm6 libxext6 "
        "&& python3 -m pip install --upgrade --no-cache-dir pip wheel setuptools packaging "
        "&& rm -rf /var/lib/apt/lists/*"
    )
    # Install torch first so safetensors detects it at build time.
    .uv_pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
    )
    .uv_pip_install(
        "lerobot==0.4.2",
        "feetech-servo-sdk",
        "huggingface-hub==0.35.3",
        "safetensors[torch]==0.7.0",
        "h5py==3.15.1",
        "pandas==2.3.3",
        "numpy==2.2.6",
        "tqdm==4.67.1",
    )
    .add_local_file("config.py", remote_path="/root/config.py")
)

train_image = (
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
        "wandb>=0.12.10",
        "deepspeed",
    )
    .add_local_dir("diffsynth", remote_path="/root/diffsynth", copy=True)
    .add_local_file(
        "train_wan_t2v_act_embed.py",
        remote_path="/root/train_wan_t2v_act_embed.py",
        copy=True,
    )
    .add_local_file("config.py", remote_path="/root/config.py")
)

prep_image = (
    modal.Image.debian_slim(python_version="3.10")
    .uv_pip_install(
        "h5py==3.15.1",
        "Pillow==11.1.0",
    )
    .add_local_file("config.py", remote_path="/root/config.py")
)

infer_image = (
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
    )
    .add_local_dir("diffsynth", remote_path="/root/diffsynth", copy=True)
    .add_local_file(
        "inference.py",
        remote_path="/root/inference.py",
        copy=True,
    )
    .add_local_file("config.py", remote_path="/root/config.py")
)

vol = modal.Volume.from_name("my-volume", create_if_missing=True, version=2)


num_cpus = 8.0
timeout = 60 * 60 * 8  # 8 hours

data_dir = "data/act_dataset"
fps = 30.0
