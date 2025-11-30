import modal

from config import infer_image, mount_path, num_cpus, timeout, vol

app = modal.App("run-inference")


@app.function(
    image=infer_image,
    volumes={mount_path: vol},
    cpu=num_cpus,
    gpu="A100-80GB",
    timeout=timeout,
)
def run_inference(model_name: str):
    import subprocess
    import sys

    cmd = [
        "python",
        "inference.py",
        "--meta_path",
        f"{str(mount_path)}/data/act_dataset/inference/metadata.json",
        "--output_subdir",
        "finetune" if model_name else "base",
        "--action",
        "--action_alpha",
        "0.3",
        "--action_dim",
        "384",
        "--action_encoded_path",
        f"{str(mount_path)}/data/act_dataset/train/all_actions.pt",
    ]

    if model_name:
        cmd.append("--lora_path")
        cmd.append(
            f"{str(mount_path)}/weights/checkpoints/{model_name}/checkpoint/mp_rank_00_model_states.pt"
        )
        print(f"Running inference with model {model_name}")
    else:
        print("Running inference with base world model")

    subprocess.run(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True,
    )


@app.local_entrypoint()
def main(model_name: str = None):
    run_inference.remote(model_name)
