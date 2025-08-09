# CUstom-Face-blur-ps
A production-ready script to apply oval face blurring on all videos in a folder using Ultralytics YOLOv8n-face and OpenCV.


## Features

- Batch video processing from a folder
- Face detection with YOLOv8n-face
- Elliptical blur applied to face regions
- Output maintains original resolution, codec, and FPS
- Shell launcher with argument parsing
- Auto model download if not found
- GPU support if available

## Usage

```bash
chmod +x run_face_blur.sh
./run_face_blur.sh --input_folder /path/to/videos --output_folder /path/to/output
```

## Setup

```bash
pip install -r requirements.txt
```

## Model

Uses [YOLOv8n-face](https://github.com/ultralytics/ultralytics) pretrained weights.
