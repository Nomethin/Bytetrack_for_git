import os
import json
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm  # 用于可视化进度条

# Paths
DATA_PATH = r"D:\bytetrack\datasets\stanford_drone"  # 修改为您的数据集路径
VIDEOS_PATH = os.path.join(DATA_PATH, "videos")
ANNOTATIONS_PATH = os.path.join(DATA_PATH, "annotations")
OUT_PATH = os.path.join(DATA_PATH, "processed")
FRAMES_PATH = os.path.join(OUT_PATH, "frames")
COCO_PATH = os.path.join(OUT_PATH, "annotations")

# Create directories
os.makedirs(FRAMES_PATH, exist_ok=True)
os.makedirs(COCO_PATH, exist_ok=True)

def extract_frames(video_path, output_dir):
    """
    Extract frames from a video file.
    :param video_path: Path to the video file.
    :param output_dir: Directory to save extracted frames.
    :return: Dictionary mapping frame index to file path.
    """
    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    frame_map = {}
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数
    with tqdm(total=total_frames, desc=f"Extracting frames: {os.path.basename(video_path)}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_file = os.path.join(output_dir, f"{frame_index:06d}.jpg")
            cv2.imwrite(frame_file, frame)
            frame_map[frame_index] = frame_file
            frame_index += 1
            pbar.update(1)
    cap.release()
    return frame_map

def convert_to_coco(videos_path, annotations_path, frames_path, coco_path):
    """
    Convert Stanford Drone Dataset structure to COCO format.
    :param videos_path: Path to the videos directory.
    :param annotations_path: Path to the annotations directory.
    :param frames_path: Path to save extracted frames.
    :param coco_path: Path to save COCO JSON file.
    """
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    category_mapping = defaultdict(lambda: len(category_mapping) + 1)
    image_id = 0
    annotation_id = 0

    # Process each category (bookstore, little, etc.)
    for category in tqdm(os.listdir(videos_path), desc="Processing categories"):
        category_videos_path = os.path.join(videos_path, category)
        category_annotations_path = os.path.join(annotations_path, category)

        if not os.path.isdir(category_videos_path) or not os.path.isdir(category_annotations_path):
            continue

        # Process each video in the category
        for video_folder in tqdm(os.listdir(category_videos_path), desc=f"Processing videos in {category}"):
            video_file = os.path.join(category_videos_path, video_folder, "video.mov")
            annotation_file = os.path.join(category_annotations_path, video_folder, "annotations.txt")

            if not os.path.exists(video_file):
                print(f"Video file not found: {video_file}. Skipping...")
                continue

            if not os.path.exists(annotation_file):
                print(f"Annotation file not found: {annotation_file}. Skipping...")
                continue

            video_frames_path = os.path.join(frames_path, category, video_folder)
            os.makedirs(video_frames_path, exist_ok=True)

            # Extract frames
            frame_map = extract_frames(video_file, video_frames_path)

            # Process annotations
            with open(annotation_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    track_id, xmin, ymin, xmax, ymax, frame, lost, occluded, generated, label = parts

                    # Skip lost or generated annotations
                    if int(lost) == 1 or int(generated) == 1:
                        continue

                    frame_index = int(frame)

                    if frame_index not in frame_map:
                        print(f"Frame {frame_index} not found in video {video_folder}. Skipping...")
                        continue

                    # Add image entry to COCO
                    frame_file = frame_map[frame_index]
                    if frame_file not in [img["file_name"] for img in coco_format["images"]]:
                        coco_format["images"].append({
                            "id": image_id,
                            "file_name": frame_file.replace(DATA_PATH + "\\", ""),
                            "height": 1080,  # Placeholder
                            "width": 1920,   # Placeholder
                            "frame_id": frame_index,
                            "video_id": hash(f"{category}_{video_folder}")
                        })
                        image_id += 1

                    # Add annotation entry to COCO
                    coco_format["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id - 1,
                        "category_id": category_mapping[label.strip('"')],
                        "bbox": [int(xmin), int(ymin), int(xmax) - int(xmin), int(ymax) - int(ymin)],
                        "area": (int(xmax) - int(xmin)) * (int(ymax) - int(ymin)),
                        "iscrowd": 0,
                        "track_id": int(track_id)
                    })
                    annotation_id += 1

    # Add categories
    coco_format["categories"] = [{"id": id_, "name": name} for name, id_ in category_mapping.items()]

    # Save COCO JSON file
    output_file = os.path.join(coco_path, "stanford_drone_dataset_coco.json")
    with open(output_file, "w") as json_file:
        json.dump(coco_format, json_file, indent=4)

    print(f"COCO format annotations saved to {output_file}")

if __name__ == "__main__":
    convert_to_coco(VIDEOS_PATH, ANNOTATIONS_PATH, FRAMES_PATH, COCO_PATH)
    