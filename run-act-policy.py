#!/usr/bin/env python3
"""Extract ACT latent action embeddings for converted HDF5 trajectories.

This utility loads the pretrained ACT policy published at https://huggingface.co/aphamm/act,
feeds each HDF5 trajectory through the policy, captures the transformer decoder activations
right before the final linear ``action_head`` layer, and stores the resulting 384-dimensional
latent action vectors in a ``.pt`` file that ``train_wan_t2v_act_embed.py`` can ingest.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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


def _convert_feature_dict(
    raw: Dict[str, Dict[str, object]],
) -> Dict[str, PolicyFeature]:
    features: Dict[str, PolicyFeature] = {}
    for name, cfg in raw.items():
        feature_type = FeatureType[str(cfg["type"])]
        shape = tuple(int(dim) for dim in cfg["shape"])
        features[name] = PolicyFeature(type=feature_type, shape=shape)
    return features


def _load_act_config(model_id: str, device: str) -> ACTConfig:
    config_path = hf_hub_download(
        repo_id=model_id,
        filename="config.json",
    )
    with open(config_path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)

    payload.pop("type", None)
    payload["input_features"] = _convert_feature_dict(payload["input_features"])
    payload["output_features"] = _convert_feature_dict(payload["output_features"])
    payload["normalization_mapping"] = {
        key: NormalizationMode[value]
        for key, value in payload.get("normalization_mapping", {}).items()
    }
    payload["device"] = device
    return ACTConfig(**payload)


def _load_policy(model_id: str, config: ACTConfig, device: torch.device) -> ACTPolicy:
    policy = ACTPolicy.from_pretrained(
        pretrained_name_or_path=model_id,
        config=config,
    )
    policy.to(device)
    policy.eval()
    return policy


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
    state = torch.from_numpy(frame.astype(np.float32)).unsqueeze(0)
    return state


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


def _iter_hdf5_chunks(
    path: Path,
    chunk_size: int,
) -> Iterable[Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor, int]]:
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


def extract_latents_for_file(
    file_path: Path,
    policy: ACTPolicy,
    chunk_size: int,
    device: torch.device,
    hook: _ActionHeadHook,
) -> np.ndarray:
    latent_segments: List[np.ndarray] = []
    # Iterate chunk-by-chunk so every frame in the trajectory contributes to the final latent tensor.
    for state, images, actions, action_pad, valid_len in _iter_hdf5_chunks(
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


def resolve_file_paths(metadata: pd.DataFrame, dataset_dir: Path) -> List[Path]:
    resolved = []
    for _, row in metadata.iterrows():
        raw_path = row.get("file_path")
        if isinstance(raw_path, str) and raw_path:
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = (dataset_dir / raw_path).resolve()
        else:
            filename = row["file_name"]
            candidate = dataset_dir / "episodes" / str(filename)
        resolved.append(candidate)
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract ACT latent action embeddings from HDF5 files."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/act_dataset"))
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/act_dataset/train/all_actions.pt"),
    )
    parser.add_argument("--model-id", type=str, default="aphamm/act")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--max-files",
        type=int,
        default=1,
        help="Limit number of files to process (default: 1)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override model chunk size if provided",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.set_grad_enabled(False)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    config = _load_act_config(args.model_id, args.device)
    if args.chunk_size is not None:
        config = replace(
            config,
            chunk_size=args.chunk_size,
            n_action_steps=min(args.chunk_size, config.n_action_steps),
        )
    chunk_size = config.chunk_size

    policy = _load_policy(args.model_id, config, device)
    hook = _ActionHeadHook()
    handle = policy.model.action_head.register_forward_pre_hook(hook)

    try:
        metadata_path = args.metadata or args.dataset_dir / "metadata.csv"
        metadata = pd.read_csv(metadata_path)
        file_paths = resolve_file_paths(metadata, args.dataset_dir)

        processed_paths: List[str] = []
        encoded_actions: List[torch.Tensor] = []

        for idx, path in enumerate(
            tqdm(file_paths, total=len(file_paths), desc="files")
        ):
            if args.max_files is not None and idx >= args.max_files:
                break
            path = Path(path)
            if not path.exists():
                print(f"Skipping missing file: {path}")
                continue
            latents = extract_latents_for_file(path, policy, chunk_size, device, hook)
            # Replace /Users/pham/Documents/world-eval with /mnt in the path if present
            str_path = str(path)
            if str_path.startswith("/Users/pham/Documents/world-eval"):
                mnt_path = str_path.replace("/Users/pham/Documents/world-eval", "/mnt")
                path = Path(mnt_path)
            processed_paths.append(str(path))
            encoded_actions.append(torch.from_numpy(latents))

        if not processed_paths:
            raise RuntimeError(
                "No HDF5 files were processed. Check dataset paths and metadata."
            )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"file_path": processed_paths, "encoded_action": encoded_actions},
            args.output,
        )
        tensor_shapes = [tuple(t.shape) for t in encoded_actions]
        print(f"Encoded tensor shapes: {tensor_shapes}")
        print(f"Processed paths: {processed_paths}")
        print(
            f"Saved latent actions for {len(processed_paths)} file(s) to {args.output}"
        )
    finally:
        handle.remove()


if __name__ == "__main__":
    main()
