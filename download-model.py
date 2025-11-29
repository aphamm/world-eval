"""
Download Wan-AI/Wan2.1-I2V-14B-480P model to Modal DFS volume.
"""

import modal

from config import hf_image, mount_path, num_cpus, timeout, vol

app = modal.App("download-model")


@app.function(image=hf_image, volumes={mount_path: vol}, cpu=num_cpus, timeout=timeout)
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
