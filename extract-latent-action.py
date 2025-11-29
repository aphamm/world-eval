#!/usr/bin/env python3
"""Extract ACT latent action embeddings on Modal DFS.

This utility loads a pretrained ACT policy on Hugging Face, feeds each HDF5 trajectory
through the policy, captures the latent action vectors right before the final linear layer
and stores the resulting 384-dimensional ``.pt`` file that ``train_wan_t2v_act_embed.py``
understands.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import modal

from config import act_image, mount_path, num_cpus, timeout, vol

app = modal.App("extract-latent-action")


@app.function(
    image=act_image,
    volumes={mount_path: vol},
    timeout=timeout,
    cpu=num_cpus,
    gpu="L40S",
)
def extract_latent_actions(hf_model: str):
    import json

    import h5py
    import numpy as np
    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
    from tqdm import tqdm

    from config import data_dir

    def _convert_feature_dict(
        raw: Dict[str, Dict[str, object]],
    ) -> Dict[str, PolicyFeature]:
        features: Dict[str, PolicyFeature] = {}
        for name, cfg in raw.items():
            feature_type = FeatureType[str(cfg["type"])]
            shape = tuple(int(dim) for dim in cfg["shape"])
            features[name] = PolicyFeature(type=feature_type, shape=shape)
        return features

    def _load_act_config(model_id: str, device: torch.device) -> ACTConfig:
        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(config_path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)

        payload.pop("type", None)
        payload["input_features"] = _convert_feature_dict(payload["input_features"])
        payload["output_features"] = _convert_feature_dict(payload["output_features"])
        payload["normalization_mapping"] = {
            key: NormalizationMode[value]
            for key, value in payload.get("normalization_mapping", {}).items()
        }
        payload["device"] = str(device)
        return ACTConfig(**payload)

    class _ActionHeadHook:
        def __init__(self) -> None:
            self._buffer: torch.Tensor | None = None

        def __call__(self, _module, inputs) -> None:
            tensor = inputs[0]
            self._buffer = tensor.detach().clone()

        @property
        def data(self) -> torch.Tensor:
            if self._buffer is None:
                raise RuntimeError("Latent buffer is empty. Did the model run?")
            return self._buffer

        def reset(self) -> None:
            self._buffer = None

    def _prepare_image(frame: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(frame.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        return tensor

    def _prepare_state(frame: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(frame.astype(np.float32)).unsqueeze(0)

    def _prepare_actions(
        chunk: np.ndarray, chunk_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        chunk_len = chunk.shape[0]
        pad_len = max(0, chunk_size - chunk_len)
        action_tensor = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0)
        if pad_len:
            padding = torch.zeros(
                (1, pad_len, action_tensor.shape[-1]), dtype=action_tensor.dtype
            )
            action_tensor = torch.cat([action_tensor, padding], dim=1)
        action_is_pad = torch.zeros((1, chunk_size), dtype=torch.bool)
        if pad_len:
            action_is_pad[0, -pad_len:] = True
        return action_tensor, action_is_pad, chunk_len

    def _iter_trajectories(
        path: Path,
        chunk_size: int,
    ) -> Iterable[
        Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor, int]
    ]:
        with h5py.File(path, "r") as handle:
            actions = handle["action"][:]
            joints = handle["observations"]["joint_positions"][:]
            cam_top = handle["observations"]["images"].get("cam_top")
            cam_front = handle["observations"]["images"].get("cam_front")
            if cam_top is None and cam_front is None:
                raise ValueError(f"No RGB streams found in {path}")

            num_frames = actions.shape[0]
            for start in range(0, num_frames, chunk_size):
                stop = min(start + chunk_size, num_frames)
                action_tensor, action_pad, chunk_len = _prepare_actions(
                    actions[start:stop], chunk_size
                )
                state_tensor = _prepare_state(joints[start])

                images: List[torch.Tensor] = []
                if cam_top is not None:
                    images.append(_prepare_image(cam_top[start]))
                if cam_front is not None:
                    images.append(_prepare_image(cam_front[start]))

                yield state_tensor, images, action_tensor, action_pad, chunk_len

    def extract_latent(
        file_path: Path,
        policy: ACTPolicy,
        chunk_size: int,
        device: torch.device,
        hook: _ActionHeadHook,
    ) -> np.ndarray:
        latent_segments: List[np.ndarray] = []
        for state, images, actions, action_pad, valid_len in _iter_trajectories(
            file_path, chunk_size
        ):
            batch = {
                OBS_STATE: state.to(device),
                OBS_IMAGES: [img.to(device) for img in images],
                ACTION: actions.to(device),
                "action_is_pad": action_pad.to(device),
            }
            hook.reset()
            with torch.no_grad():
                _ = policy.model(batch)
            latent_chunk = hook.data[0, :valid_len].cpu().numpy()
            latent_segments.append(latent_chunk)
        return np.vstack(latent_segments)

    torch.set_grad_enabled(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = _load_act_config(hf_model, device)
    policy = ACTPolicy.from_pretrained(
        pretrained_name_or_path=hf_model,
        config=config,
    )
    policy.to(device)
    policy.eval()
    hook = _ActionHeadHook()
    handle = policy.model.action_head.register_forward_pre_hook(hook)
    data_dir = Path(mount_path) / data_dir

    try:
        metadata = pd.read_csv(data_dir / "metadata.csv")
        file_paths = [row for row in metadata["file_path"]]

        processed_paths: List[str] = []
        encoded_actions: List[torch.Tensor] = []

        for path in tqdm(file_paths, total=len(file_paths), desc="files"):
            latents = extract_latent(path, policy, config.chunk_size, device, hook)
            processed_paths.append(str(path))
            encoded_actions.append(torch.from_numpy(latents))

        if not processed_paths:
            raise RuntimeError(
                "No HDF5 files were processed. Check dataset paths and metadata."
            )

        output_dir = Path(mount_path) / data_dir / "train"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "all_actions.pt"
        torch.save(
            {"file_path": processed_paths, "encoded_action": encoded_actions},
            output_path,
        )
        tensor_shapes = [tuple(t.shape) for t in encoded_actions]
        print(f"Encoded tensor shapes: {tensor_shapes}")
        print(f"Processed paths: {processed_paths[:5]}")
        print(
            f"Saved latent actions for {len(processed_paths)} file(s) to {output_path}"
        )
    finally:
        handle.remove()


@app.local_entrypoint()
def main(hf_model: str = "aphamm/act"):
    extract_latent_actions.remote(hf_model=hf_model)
