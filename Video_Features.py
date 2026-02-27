import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
import argparse
from tqdm import tqdm
import glob

INPUT_SHAPE = (299, 299, 3)
BATCH_SIZE = 64
WINDOW_LENGTH_FRAMES = 5400
OVERLAP_RATIO = 0.90

def generate_sliding_windows(data, win_len, overlap_ratio):
    n_frames = data.shape[0]
    if n_frames < win_len:
        return None
    step_size = int(win_len * (1 - overlap_ratio))
    if step_size < 1:
        step_size = 1
    windows = []
    for start_idx in range(0, n_frames - win_len + 1, step_size):
        end_idx = start_idx + win_len
        window = data[start_idx:end_idx]
        windows.append(window)
    if not windows:
        return None
    return np.array(windows)

class VideoFeatureExtractor:
    def __init__(self):
        self.model = InceptionResNetV2(
            weights='imagenet',
            include_top=False,
            pooling='avg',
            input_shape=INPUT_SHAPE
        )
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (INPUT_SHAPE[0], INPUT_SHAPE[1]))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        return face_img

    def extract_features(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        batch_frames = []
        features_list = []
        pbar = tqdm(total=total_frames)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            face_tensor = self.preprocess_frame(frame)
            if face_tensor is not None:
                batch_frames.append(face_tensor)
            else:
                batch_frames.append(np.zeros(INPUT_SHAPE, dtype=np.uint8))
            
            if len(batch_frames) == BATCH_SIZE:
                batch_arr = np.array(batch_frames, dtype=np.float32)
                batch_arr = preprocess_input(batch_arr)
                batch_features = self.model.predict(batch_arr, verbose=0)
                features_list.append(batch_features)
                batch_frames = []
            pbar.update(1)
            
        if len(batch_frames) > 0:
            batch_arr = np.array(batch_frames, dtype=np.float32)
            batch_arr = preprocess_input(batch_arr)
            batch_features = self.model.predict(batch_arr, verbose=0)
            features_list.append(batch_features)
        
        pbar.close()
        cap.release()
        
        if not features_list:
            return None
        
        X = np.concatenate(features_list, axis=0)
        return generate_sliding_windows(X, WINDOW_LENGTH_FRAMES, OVERLAP_RATIO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    extractor = VideoFeatureExtractor()
    video_files = glob.glob(os.path.join(args.input_dir, "*.*"))
    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in video_files if f.lower().endswith(valid_extensions)]

    print(f"Found {len(video_files)} videos.")

    for video_path in video_files:
        try:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(args.output_dir, f"{base_name}_visual_windows.npy")
            
            if os.path.exists(output_path):
                print(f"Skipping {base_name}")
                continue

            windows = extractor.extract_features(video_path)
            
            if windows is not None:
                np.save(output_path, windows)
                print(f"Saved: {output_path} | Shape: {windows.shape}")
            else:
                print(f"Skipped {base_name}: Insufficient frames")
                
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")

if __name__ == "__main__":
    main()
