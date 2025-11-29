# WorldEval w/ ACT LeRobot Dataset

### Packages

```bash
# install packages
uv venv .venv --python=3.10
source .venv/bin/activate
uv pip install -r requirements.txt
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
```

You need to organize the HDF5 files containing the robot trajectory data as follows:

```bash
act_dataset/
├── episodes/
  ├── episode_0001.h5
  ├── episode_0002.h5
  └── episode_0003.h5
├── train (empty dir to store processed tensor)
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

Wan-Video is a collection of video synthesis models open-sourced by Alibaba. Download the weights [14B image-to-video 480P model](https://modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P). Download models using the huggingface CLI.

```bash
hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./models
```

```bash
modal volume rm my-volume data/act_dataset/train/all_actions.pt
modal volume put my-volume data/act_dataset/train/all_actions.pt data/act_dataset/train/all_actions.pt
```

### Step 2: Extract Action Latents

The weights of the VLA policy used in our paper: [ACT](https://huggingface.co/aphamm/act) with `dim_model = 384`.

```bash
modal volume create my-volume --version=2
modal volume put my-volume data
modal run --detach extract-latents.py
```

After, cached files will be stored in the dataset folder.

```bash
act_dataset/
├── episodes/
├── train
    ├── file1.hdf5.tensors.pth
    └── file2.hdf5.tensors.pth
└── metadata.csv
```

### Step 3: Finetune World Model with LoRA

```bash
wanvideo/scripts/lora_finetune.sh
```

Note: separate the safetensor files with a comma.

## Running World Model Inference

### Step 1: Extract Action Embeddings

Extract 384-dimensional latent action embeddings using the ACT policy checkpoint (`dim_model = 384`). Save them in a `.pt` file with the following structure:

```json
{
  "file_path": ["path/to/file1.hdf5", "path/to/file2.hdf5"],
  "encoded_action": [latent_action_vector1, latent_action_vector2]
}
```

The ACT checkpoint we rely on is available on [Hugging Face](https://huggingface.co/aphamm/act); please follow its README to export latent actions compatible with this repository.

### Step 2: Sample Frames

Use `utils/sample_frames_from_dir_for_test` to extract sample frames from the HDF5 file for testing; this will generate a `metadata.json` file and save the first frame for use in generation.

### Step 3: Run Inference Script

```bash
wanvideo/scripts/inference.sh
```

#### Acknowledgement

We build our project based on:

- [WAN2.1](https://github.com/Wan-Video/Wan2.1): a comprehensive and open suite of video foundation models that pushes the boundaries of video generation.
- [DiffSynth Studio](https://github.com/modelscope/DiffSynth-Studio): an open-source project aimed at exploring innovations in AIGC technology, licensed under the Apache License 2.0.
- [WorldEval](https://github.com/liyaxuanliyaxuan/Worldeval?tab=readme-ov-file): finetuning a video generation model into a world simulator that follows latent action to generate the robot video.
