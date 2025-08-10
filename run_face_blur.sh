#!/bin/bash

set -e

print_usage() {
  echo "Usage: $0 --input_folder /path/to/videos --output_folder /path/to/output"
}

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --input_folder)
      INPUT_FOLDER="$2"
      shift 2
      ;;
    --output_folder)
      OUTPUT_FOLDER="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      print_usage
      exit 1
      ;;
  esac
done

if [[ -z "$INPUT_FOLDER" || -z "$OUTPUT_FOLDER" ]]; then
  echo "Input and output folder arguments are required."
  print_usage
  exit 1
fi

if [[ ! -d "$INPUT_FOLDER" ]]; then
  echo "Error: Input folder $INPUT_FOLDER does not exist."
  exit 1
fi

mkdir -p "$OUTPUT_FOLDER"

if ! command -v yolo &> /dev/null; then
  echo "Installing Python dependencies..."
  pip install -r requirements.txt
fi

python3 face_blur_pipeline.py --input "$INPUT_FOLDER" --output "$OUTPUT_FOLDER"
