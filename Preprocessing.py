# Ensure required packages are installed:
# pip install torch torchvision facenet-pytorch opencv-python tqdm matplotlib

import glob
import os
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from tqdm import tqdm

# Change the path to the directory where your videos are stored
video_files = glob.glob(r'C:/Users/Hamid/PycharmProjects/pythonProject/Dataset/*.mp4')
frame_count = []

# Ensure the system is using GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize MTCNN for GPU face detection
mtcnn = MTCNN(keep_all=True, device=device)

# Calculate average frame count and remove videos with less than 150 frames
valid_video_files = []
for video_file in video_files:
    cap = cv2.VideoCapture(video_file)
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_total >= 150:
        frame_count.append(frame_total)
        valid_video_files.append(video_file)
    cap.release()

print("Frames:", frame_count)
print("Total number of valid videos:", len(valid_video_files))
print('Average frames per video:', np.mean(frame_count))

# Function to extract frames from video
def frame_extract(path):
    vidObj = cv2.VideoCapture(path)
    while True:
        success, image = vidObj.read()
        if not success:
            break
        if image is None or image.size == 0:
            continue
        yield image
    vidObj.release()

# Create directory for output if it doesn't exist
output_dir = 'C:/Users/Hamid/PycharmProjects/pythonProject/preprocessed'
os.makedirs(output_dir, exist_ok=True)

# Function to create face-only videos
def create_face_videos_with_mtcnn(path_list, out_dir):
    already_present_count = len(glob.glob(out_dir + '/*.mp4'))
    print("Number of videos already present:", already_present_count)

    for path in tqdm(path_list):
        out_path = os.path.join(out_dir, os.path.basename(path))
        if os.path.exists(out_path):
            print("File already exists:", out_path)
            continue

        # Open video writer for the output video
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mpv4'), 30, (112, 112))

        frame_idx = 0
        for idx, frame in enumerate(frame_extract(path)):
            # Limit frames for quicker processing and skip processing after 150 frames
            if frame_idx > 150:
                break
            frame_idx += 1

            try:
                # Convert BGR (OpenCV) to RGB (required by MTCNN)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes, _ = mtcnn.detect(frame_rgb)

                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = [int(coord) for coord in box]
                        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                            continue  # Skip invalid box coordinates

                        cropped_face = frame[y1:y2, x1:x2]
                        if cropped_face.size == 0:  # Skip if the cropped frame is empty
                            continue

                        resized_face = cv2.resize(cropped_face, (112, 112))
                        out.write(resized_face)
            except cv2.error as e:
                print(f"OpenCV error processing frame {idx} in {path}: {e}")
            except Exception as e:
                print(f"General error processing frame {idx} in {path}: {e}")
                continue

        out.release()
        print(f"Finished processing: {out_path}")

create_face_videos_with_mtcnn(valid_video_files, output_dir)
