import json
from pathlib import Path

import h5py
import modal
from PIL import Image

from config import mount_path, num_cpus, prep_image, timeout, vol

app = modal.App("prepare-inference")


@app.function(
    image=prep_image,
    volumes={mount_path: vol},
    cpu=num_cpus,
    timeout=timeout,
)
def prepare_inference():
    """Process all HDF5 files in the input directory"""

    from config import data_dir, mount_path

    def get_text(f, frame_id):
        text = "No text description."
        if "language_raw" in f:
            text = f["language_raw"][0].decode("utf-8").strip()
        return text

    # Use modal mount path
    data_dir = Path(mount_path) / data_dir

    # Create output directory
    output_dir = Path(data_dir) / "inference"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = []

    # Get all HDF5 files in input directory with index >= 51
    episode_dir = Path(data_dir) / "episodes"
    files = list(episode_dir.glob("*.h5"))

    print(f"Found {len(files)} HDF5 files to process...")

    for file_path in files:
        try:
            with h5py.File(file_path, "r") as f:
                # Get top camera frames
                frames = f["observations/images/cam_top"]

                # Get first frame
                frame_idx = 0
                frame = frames[frame_idx]
                text = frame["language_raw"][0].decode("utf-8").strip()

                # Save image
                image_filename = f"{file_path.stem}_frame_{frame_idx}.png"
                image_path = output_dir / image_filename
                Image.fromarray(frame).save(image_path)

                metadata.append(
                    {
                        "image_path": str(image_path),
                        "file_path": str(file_path),
                        "frame_index": int(frame_idx),
                        "language": text,
                    }
                )

                print(f"Processed {file_path.name} - frame {frame_idx}")

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    # Save metadata to JSON
    json_path = output_dir / "metadata.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nCompleted! Saved {len(metadata)} images and metadata to {output_dir}")


@app.local_entrypoint()
def main():
    prepare_inference.remote()
