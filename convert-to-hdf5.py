"""
Convert a LeRobot parquet/video dataset into ACT-style HDF5 files using Modal.
"""

import pathlib

import modal

app = modal.App("convert-to-hdf5")
vol = modal.Volume.from_name("my-volume", create_if_missing=True, version=2)
mount_path = pathlib.Path("/")


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .uv_pip_install("lerobot", "opencv-python", "h5py")
)


@app.function(image=image, volumes={mount_path: vol}, timeout=60 * 60 * 2, cpu=8.0)
def run_conversion(
    dataset_name: str = "aphamm/real-teleop-v0",
    split: str = "train",
    dataset_subdir: str = "act_dataset",
    fps: float = 30.0,
):
    """Modal-hosted conversion that persists episodes on the DFS volume."""
    import tempfile
    from typing import Any, Dict, List, Tuple

    import cv2
    import h5py
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from tqdm import tqdm

    def fetch_feature(row: Dict, dotted_key: str):
        """
        Retrieves a possibly nested feature (e.g. 'observation.images.top')
        from a dataset row.
        """
        if dotted_key in row:
            return row[dotted_key]

        current = row
        for part in dotted_key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                available = (
                    list(current.keys())
                    if isinstance(current, dict)
                    else f"type={type(current).__name__}"
                )
                raise KeyError(
                    f"Key '{part}' missing while resolving '{dotted_key}'. "
                    f"Available keys: {available}"
                )
        return current

    def resolve_video_path(video_feature) -> str:
        """
        Hugging Face Video features are returned either as a path string or
        a dict containing 'path' / 'bytes'. Normalize to a local path.
        """
        if isinstance(video_feature, dict):
            if video_feature.get("path"):
                return video_feature["path"]
            if video_feature.get("bytes") is not None:
                # Write temporary file
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp_file.write(video_feature["bytes"])
                tmp_file.flush()
                tmp_file.close()
                return tmp_file.name
        return str(video_feature)

    def load_video_frames(path: str) -> np.ndarray:
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        if not frames:
            raise RuntimeError(f"Failed to decode frames from {path}")
        return np.array(frames, dtype=np.uint8)

    def frame_from_row(
        row: Dict,
        video_entry,
        cache: Dict[str, np.ndarray],
        chunk_size: int,
    ) -> Tuple[np.ndarray, str]:
        path = resolve_video_path(video_entry)
        if path not in cache:
            cache[path] = load_video_frames(path)
        frames = cache[path]

        frame_index = int(row.get("frame_index", 0))
        chunk_index = int(row.get("chunk_index", 0))
        # Align frame index to chunk-local index
        local_index = frame_index - chunk_index * chunk_size
        if local_index < 0 or local_index >= len(frames):
            # Fall back to modulo to stay in range
            local_index = frame_index % len(frames)
        return frames[local_index], path

    def to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            value = value.detach().cpu().numpy()
        elif hasattr(value, "cpu") and hasattr(value, "numpy"):
            value = value.cpu().numpy()
        elif isinstance(value, list):
            value = np.asarray(value)
        elif isinstance(value, tuple):
            value = np.asarray(value)
        elif not isinstance(value, np.ndarray):
            value = np.asarray(value)
        return value

    def to_image_array(value: Any) -> np.ndarray:
        arr = to_numpy(value)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[-1]:
            arr = np.transpose(arr, (1, 2, 0))
        return arr.astype(np.uint8)

    def load_lerobot_dataset(dataset_name) -> Dict[int, List[Dict]]:
        dataset = LeRobotDataset(repo_id=dataset_name)
        episodes: Dict[int, List[Dict]] = {}
        for idx in tqdm(
            range(len(dataset)), desc="Decoding frames with LeRobotDataset"
        ):
            item = dataset[idx]
            normalized: Dict[str, Any] = {}
            for key, value in item.items():
                normalized[key] = to_numpy(value)
            # Ensure scalar metadata are plain ints
            for scalar_key in ["episode_index", "frame_index", "task_index"]:
                if scalar_key in normalized:
                    normalized[scalar_key] = int(
                        np.asarray(normalized[scalar_key]).item()
                    )
            episodes.setdefault(normalized["episode_index"], []).append(normalized)
        print(f"Found {len(episodes)} episodes via LeRobotDataset.")
        return episodes

    def write_episode(
        ep_idx: int,
        actions: List[np.ndarray],
        joint_pos: List[np.ndarray],
        cam_high_frames: List[np.ndarray],
        cam_front_frames: List[np.ndarray],
        ep_rows: List[Dict],
        dataset_dir: pathlib.Path,
        fps: float,
    ):
        actions_arr = np.stack(actions).astype(np.float32)
        joint_arr = np.stack(joint_pos).astype(np.float32)
        qpos = joint_arr.copy()
        qvel = np.zeros_like(qpos)
        if len(qpos) > 1:
            qvel[1:] = (qpos[1:] - qpos[:-1]) * fps
            qvel[0] = qvel[1]

        cam_high = np.stack(cam_high_frames).astype(np.uint8)
        cam_front = np.stack(cam_front_frames).astype(np.uint8)

        task_name = ep_rows[0].get("task")
        assert task_name is not None, "Task name is required"
        language = str(task_name).encode("utf-8")

        out_path = dataset_dir / "episodes" / f"episode_{ep_idx:04d}.h5"
        with h5py.File(out_path, "w") as f:
            f.create_dataset("action", data=actions_arr, compression="gzip")
            f.create_dataset(
                "language_raw",
                data=np.array([language]),
                dtype="S{}".format(len(language)),
            )
            obs_group = f.create_group("observations")
            obs_group.create_dataset(
                "joint_positions", data=joint_arr, compression="gzip"
            )
            obs_group.create_dataset("qpos", data=qpos, compression="gzip")
            obs_group.create_dataset("qvel", data=qvel, compression="gzip")
            img_group = obs_group.create_group("images")
            img_group.create_dataset("cam_top", data=cam_high, compression="gzip")
            img_group.create_dataset("cam_front", data=cam_front, compression="gzip")
        print(f"Wrote {out_path}")

    def write_metadata(dataset_dir: pathlib.Path, episodes):
        meta_path = dataset_dir / "metadata.csv"
        with meta_path.open("w") as f:
            f.write("file_path,file_name,text\n")
            for ep_idx in sorted(episodes.keys()):
                file_name = f"episode_{ep_idx:04d}.h5"
                file_path = dataset_dir / "episodes" / file_name
                f.write(f'{file_path},{file_name},""\n')

    dataset_dir = mount_path / dataset_subdir
    episodes_dir = dataset_dir / "episodes"
    train_dir = dataset_dir / split
    episodes_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_lerobot_dataset(
        dataset_name=dataset_name,
    )

    for ep_idx in tqdm(sorted(episodes.keys()), desc="Converting episodes"):
        ep_rows = sorted(
            episodes[ep_idx], key=lambda x: int(np.asarray(x["frame_index"]))
        )

        actions: List[np.ndarray] = []
        joint_pos: List[np.ndarray] = []
        cam_high_frames: List[np.ndarray] = []
        cam_front_frames: List[np.ndarray] = []

        for row in ep_rows:
            actions.append(np.asarray(row["action"], dtype=np.float32))
            joint_pos.append(np.asarray(row["observation.state"], dtype=np.float32))
            top_frame = to_image_array(fetch_feature(row, "observation.images.top"))
            front_frame = to_image_array(fetch_feature(row, "observation.images.front"))

            cam_high_frames.append(top_frame)
            cam_front_frames.append(front_frame)

        write_episode(
            ep_idx,
            actions,
            joint_pos,
            cam_high_frames,
            cam_front_frames,
            ep_rows,
            dataset_dir=dataset_dir,
            fps=fps,
        )

    write_metadata(dataset_dir, episodes)
    vol.commit()


@app.local_entrypoint()
def main(
    dataset_name: str = "aphamm/real-teleop-v0",
    split: str = "train",
    dataset_subdir: str = "act_dataset",
    fps: float = 30.0,
):
    run_conversion.remote(
        dataset_name=dataset_name,
        split=split,
        dataset_subdir=dataset_subdir,
        fps=fps,
    )
