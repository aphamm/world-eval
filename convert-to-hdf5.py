"""
Convert a LeRobot parquet/video dataset into ACT-style HDF5 files using Modal.
"""

import argparse
import pathlib
from typing import Any, Dict, List

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm


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
    """
    Returns: a numpy array of shape (H, W, 3) or (H, W, 4) in uint8 format.
    """
    arr = to_numpy(value)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).round().astype(np.uint8, copy=False)
    return arr


def load_lerobot_dataset(hf_dataset) -> Dict[int, List[Dict]]:
    """
    Instantiate LeRobotDataset, convert each sample's tensor fields into Numpy arrays.
    Buckets every row into the episodes dictionary keyed by episode id.
    """
    dataset = LeRobotDataset(repo_id=hf_dataset)

    row = dataset[0]
    image = row["observation.images.top"]
    print("Image", getattr(image, "dtype"), getattr(image, "shape"))

    episodes: Dict[int, List[Dict]] = {}
    for idx in tqdm(range(len(dataset)), desc="Decoding frames"):
        item = dataset[idx]
        normalized: Dict[str, Any] = {}
        for key, value in item.items():
            normalized[key] = to_numpy(value)
        # Ensure scalar metadata are plain ints
        for scalar_key in ["episode_index", "frame_index", "task_index"]:
            if scalar_key in normalized:
                normalized[scalar_key] = int(np.asarray(normalized[scalar_key]).item())
        episodes.setdefault(normalized["episode_index"], []).append(normalized)
    print(f"Found {len(episodes)} episodes.")
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
        obs_group.create_dataset("joint_positions", data=joint_arr, compression="gzip")
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


def main():
    parser = argparse.ArgumentParser(description="Convert LeRobot dataset to HDF5.")
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default="aphamm/real-teleop-v0",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split (e.g., train, val)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/act_dataset",
        help="Directory to save the converted dataset",
    )
    parser.add_argument(
        "--fps", type=float, default=30.0, help="Frames per second for the video"
    )

    args = parser.parse_args()

    hf_dataset = args.hf_dataset
    split = args.split
    data_dir = args.data_dir
    fps = args.fps

    data_path = pathlib.Path(data_dir).expanduser()
    data_path = pathlib.Path(__file__).resolve().parent / data_path
    dataset_dir = data_path
    episodes_dir = dataset_dir / "episodes"
    train_dir = dataset_dir / split
    episodes_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_lerobot_dataset(
        hf_dataset=hf_dataset,
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
            top_frame = to_image_array(row["observation.images.top"])
            front_frame = to_image_array(row["observation.images.front"])
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


if __name__ == "__main__":
    main()
