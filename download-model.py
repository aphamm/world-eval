"""
Download Wan-AI/Wan2.1-I2V-14B-480P model to Modal DFS volume.
"""

import pathlib

import modal

app = modal.App("download-model")
vol = modal.Volume.from_name("my-volume", create_if_missing=True, version=2)
mount_path = pathlib.Path("/mnt")


image = modal.Image.debian_slim(python_version="3.10").uv_pip_install("huggingface_hub")


@app.function(image=image, volumes={mount_path: vol}, timeout=60 * 60 * 2, cpu=8.0)
def download_model():
    import subprocess

    subprocess.run(
        [
            "hf",
            "download",
            "Wan-AI/Wan2.1-I2V-14B-480P",
            "--local-dir",
            f"{str(mount_path)}/models",
        ],
        check=True,
    )
    vol.commit()


@app.local_entrypoint()
def main():
    download_model.remote()
