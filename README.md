# WorldEval w/ ACT LeRobot Dataset

### Packages

```bash
# install packages
uv venv .venv --python=3.10
source .venv/bin/activate
uv pip install -r requirements.txt
# create a modal API token
python3 -m modal setup
# create modal secret
modal secret create wandb-secret WANDB_API_KEY="<your_wandb_api_key>"
# install ffmpeg for image/video processing
brew install ffmpeg@7
# when a program needs a dynamic library at runtime, dyld searches for it in DYLD_LIBRARY_PATH
# used by developers to specify the location of custom or newly compiled libraries during development
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/ffmpeg@7/lib:$DYLD_LIBRARY_PATH
```

## Finetuning World Model

### Step 1: Prepare HDF5 Trajectory Data

```bash
python3 convert-to-hdf5.py --hf_dataset="aphamm/real-teleop-v0"
# create modal volume and upload data/ to modal DFS
modal volume create my-volume --version=2
modal volume put my-volume data data
```

You need to organize the HDF5 files containing the robot trajectory data as follows:

```bash
act_dataset/
├── episodes/
  ├── episode_0001.h5
  ├── episode_0002.h5
  └── episode_0003.h5
├── train/ (empty dir to store processed tensor)
└── metadata.csv (file_path,file_name,text)
```

where each HD5F file is structured as:

```bash
episode_0001.h5
├── action (F,6)
├── language_raw (1,)
└── observations
    ├── joint_positions (F,6)
    ├── qpos (F,6)
    ├── qvel (F,6)
    └── images
        ├── cam_top (F,480,640,3)
        └── cam_front (F,480,640,3)
```

Note: `F` indicates the number of frames in an episode.

### Step 2: Extract Action Latents

Run the Act Policy to generate action latents for all frames in the HD5F dataset. The weights of the VLA policy used in our paper: [ACT](https://huggingface.co/aphamm/act).

```bash
modal run --detach extract-latent-action.py --hf-model="aphamm/act"
```

A new `pt` file should be saved as such:

```bash
act_dataset/
├── episodes/
├── train/
    └── all_actions.pt
└── metadata.csv
```

This will extract 384-dimensional latent action embeddings and save them in a `all_actions.pt` file with the following structure:

```json
{
  "file_path": ["path/to/file1.hdf5", "path/to/file2.hdf5"],
  "encoded_action": [latent_action_vector1, latent_action_vector2]
}
```

### Step 3: Prepare Training Examples

```bash
modal run --detach generate-train-data.py
```

After, cached files will be stored in the dataset folder.

```bash
act_dataset/
├── episodes/
├── train/
    └── all_actions.pt
    ├── file1.hdf5.tensors.pth
    └── file2.hdf5.tensors.pth
└── metadata.csv
```

### Step 4: Download Model Weights

Wan-Video is a collection of video synthesis models open-sourced by Alibaba. Download the modal weights [14B image-to-video 480P model](https://modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P) to Modal DFS using the huggingface CLI.

```bash
modal run --detach download-model.py
```

### Step 5: Finetune World Model with LoRA

```bash
modal run --detach lora-finetune.py
```

## Running World Model Inference

### Step 1: Prepare Inference Data

Prepare test data for video generation inference. Create a `metadata.json` with file paths, image paths, frame indices, and language descriptions. The first frame is saved as a PNG image saved as `{filename}_frame_0.png`.

```bash
modal run --detach prepare-inference.py
```

After, the relevant files will be stored in the `dataset/inference` folder.

```bash
act_dataset/
├── episodes/
├── train/
├── inference/
    ├── metadata.json
    ├── episode_0000_frame_0.png
    └── episode_0001_frame_0.png
└── metadata.csv
```

The `metadata.json` file should look as such:

```bash
[
  {
    "image_path": ".../episode_0000_frame_0.png",
    "file_path": ".../episodes/episode_0000.h5",
    "frame_index": 0,
    "language": "Pick the small cube and put it in the box"
  },
]
```

### Step 2: Run Inference Script

```bash
modal run --detach run-inference.py --model-name="epoch=29_train_loss=0.0435.ckpt"
```

This will output generated videos in `act_dataset/inference/videos/episode_XXXX_frame_0.mp4`.

#### Acknowledgement

We build our project based on:

- [WAN2.1](https://github.com/Wan-Video/Wan2.1): a comprehensive and open suite of video foundation models that pushes the boundaries of video generation.
- [DiffSynth Studio](https://github.com/modelscope/DiffSynth-Studio): an open-source project aimed at exploring innovations in AIGC technology, licensed under the Apache License 2.0.
- [WorldEval](https://github.com/liyaxuanliyaxuan/Worldeval?tab=readme-ov-file): finetuning a video generation model into a world simulator that follows latent action to generate the robot video.
