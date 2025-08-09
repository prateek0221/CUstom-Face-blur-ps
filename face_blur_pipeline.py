import os
import cv2
import argparse
from ultralytics import YOLO
import numpy as np
from pathlib import Path
from tqdm import tqdm
import requests
import subprocess

def load_model():
    model_path = "/home/rishabh/Desktop/Prateek/face_blur_pipeline/model/yolov8m-face-lindevs.pt"
    if not os.path.exists(model_path):
        print("Downloading YOLOv8n-face model...")
        model = YOLO("https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-face.pt")
        model.save(model_path)
    else:
        model = YOLO(model_path)

    print(f"[INFO] Model loaded from: {model_path}")
    model.to("cuda")  # Use GPU for acceleration
    print("[INFO] Model moved to CUDA (GPU) for faster processing.")
    return model

def apply_oval_blur(frame, bbox):
    (x1, y1, x2, y2) = map(int, bbox)
    roi = frame[y1:y2, x1:x2]
    mask = np.zeros(roi.shape, roi.dtype)
    center = (roi.shape[1] // 2, roi.shape[0] // 2)
    axes = (roi.shape[1] // 2, roi.shape[0] // 2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 255, 255), -1)
    blurred = cv2.GaussianBlur(roi, (99, 99), 30)
    np.copyto(roi, blurred, where=mask.astype(bool))
    frame[y1:y2, x1:x2] = roi
    return frame

def convert_mov_to_mp4(input_path, converted_dir):
    converted_path = converted_dir / (input_path.stem + ".mp4")
    if not converted_path.exists():
        print(f"[INFO] Converting {input_path.name} to MP4...")
        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-vcodec", "libx264", "-acodec", "aac",
            str(converted_path)
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print(f"[ERROR] Failed to convert {input_path}")
            return None
    else:
        print(f"[INFO] Skipping conversion; MP4 already exists: {converted_path.name}")
    return converted_path

# def process_video(video_path, output_path, model):
#     print(f"[INFO] Starting processing: {video_path.name}")
#     cap = cv2.VideoCapture(str(video_path))
#     if not cap.isOpened():
#         print(f"[ERROR] Failed to open {video_path}")
#         return

#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     safe_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     if output_path.suffix.lower() == ".mov":
#         output_path = output_path.with_suffix(".mp4")

#     try:
#         orig_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
#         out = cv2.VideoWriter(str(output_path), orig_fourcc, fps, (width, height))
#         if not out.isOpened():
#             raise ValueError("Original FOURCC codec not supported. Falling back to 'mp4v'")
#     except Exception as e:
#         print(f"[WARNING] {e}")
#         out = cv2.VideoWriter(str(output_path), safe_fourcc, fps, (width, height))
#         if not out.isOpened():
#             print(f"[ERROR] Could not open output writer for {output_path}")
#             cap.release()
#             return

#     frame_idx = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         results = model.predict(frame, verbose=False)[0]
#         for box in results.boxes.xyxy:
#             frame = apply_oval_blur(frame, box)
#         out.write(frame)

#         # Checkpoint print every 100 frames
#         frame_idx += 1
#         if frame_idx % 100 == 0:
#             print(f"[INFO] Processed {frame_idx}/{frame_count} frames of {video_path.name}")

#     cap.release()
#     out.release()
#     print(f"[✅ DONE] Saved: {output_path}")

# v-2

def process_video(video_path, output_path, model):
    print(f"[INFO] Starting processing: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Failed to open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    safe_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if output_path.suffix.lower() == ".mov":
        output_path = output_path.with_suffix(".mp4")

    try:
        orig_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        out = cv2.VideoWriter(str(output_path), orig_fourcc, fps, (width, height))
        if not out.isOpened():
            raise ValueError("Original FOURCC codec not supported. Falling back to 'mp4v'")
    except Exception as e:
        print(f"[WARNING] {e}")
        out = cv2.VideoWriter(str(output_path), safe_fourcc, fps, (width, height))
        if not out.isOpened():
            print(f"[ERROR] Could not open output writer for {output_path}")
            cap.release()
            return

    frame_idx = 0
    saved_frames = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip every 5th frame unless person is detected
        if frame_idx % 5 != 0:
            frame_idx += 1
            continue
        print(f"skipped..")

        results = model.predict(frame, verbose=False)[0]
        if len(results.boxes.xyxy) > 0:
            for box in results.boxes.xyxy:
                frame = apply_oval_blur(frame, box)
            out.write(frame)
            saved_frames += 1
        # Else: no people → skip writing this frame

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"[INFO] Processed {frame_idx}/{frame_count} frames of {video_path.name}, Saved: {saved_frames}")

    cap.release()
    out.release()
    print(f"[✅ DONE] Saved {saved_frames} frames with people to: {output_path}")




def batch_process(input_dir, output_dir, model):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    converted_dir = input_dir / "converted"
    converted_dir.mkdir(exist_ok=True)

    video_files = list(input_dir.glob("*.MP4")) + list(input_dir.glob("*.avi")) + list(input_dir.glob("*.mp4"))
    if not video_files:
        print("[WARNING] No supported video files found.")
        return

    for video in tqdm(video_files, desc="Processing Videos"):
        if video.suffix.lower() == ".mov":
            converted_video = convert_mov_to_mp4(video, converted_dir)
            if converted_video is None:
                continue
            output_file = output_dir / converted_video.name
            process_video(converted_video, output_file, model)
        else:
            output_file = output_dir / video.name
            process_video(video, output_file, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input folder containing videos")
    parser.add_argument("--output", required=True, help="Output folder for blurred videos")
    args = parser.parse_args()

    model = load_model()
    batch_process(args.input, args.output, model)
