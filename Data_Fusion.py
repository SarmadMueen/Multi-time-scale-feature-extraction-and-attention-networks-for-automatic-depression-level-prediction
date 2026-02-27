import os
import numpy as np
import pandas as pd
import glob
import re
from tqdm import tqdm
import argparse


WINDOW_LENGTH_FRAMES = 5400  
OVERLAP_RATIO = 0.90         
STEP_SIZE = int(WINDOW_LENGTH_FRAMES * (1 - OVERLAP_RATIO))

def generate_sliding_windows(data_matrix, win_len, step):
    """
    Applies sliding window to the concatenated feature matrix.
    Input: (Total_Frames, Total_Features)
    Output: (N_Windows, 5400, Total_Features)
    """
    n_frames, n_feats = data_matrix.shape
    if n_frames < win_len:
        return None

    windows = []
    for start in range(0, n_frames - win_len + 1, step):
        end = start + win_len
        window = data_matrix[start:end, :]
        windows.append(window)
    
    if not windows:
        return None
    
    return np.array(windows)

def fuse_streams(visual_dir, behavioral_dir, output_file):
   
    print("--- Starting Data Fusion (Visual + Behavioral) ---")
    
    # Locate visual feature files
    visual_files = glob.glob(os.path.join(visual_dir, '*_visual_features.npy'))
    
    all_windows = []
    all_labels = []
    all_pids = []

    for v_path in tqdm(visual_files):
        # Extract PID (assuming filename format: 303_visual_features.npy)
        basename = os.path.basename(v_path)
        pid_match = re.search(r'(\d+)', basename)
        if not pid_match: continue
        pid = pid_match.group(1)
        
        # 1. Load Visual Features (Stream 1)
        # Shape: (Frames, 1536)
        X_visual = np.load(v_path)
       
        X_behavior = load_behavioral_for_pid(behavioral_dir, pid) 
        
        if X_behavior is None:
            continue

        min_len = min(X_visual.shape[0], X_behavior.shape[0])
        X_visual = X_visual[:min_len]
        X_behavior = X_behavior[:min_len]

        # X_combined shape: (Frames, 1536 + N_Behavioral)
        X_combined = np.concatenate([X_visual, X_behavior], axis=1)
        
        # 5. Sliding Window (Algorithm 1)
        windows = generate_sliding_windows(X_combined, WINDOW_LENGTH_FRAMES, STEP_SIZE)
        
        if windows is not None:
            all_windows.append(windows)
            all_pids.extend([pid] * windows.shape[0])

    if not all_windows:
        print("No valid windows generated.")
        return

    final_X = np.concatenate(all_windows, axis=0)
    final_pids = np.array(all_pids)
    
    print(f"Fusion Complete. Final Dataset Shape: {final_X.shape}")
    
    np.savez_compressed(output_file, X=final_X, pids=final_pids)
    print(f"Saved to {output_file}")

def load_behavioral_for_pid(root_path, pid):
    """
    Helper to load OpenFace CSVs and return a numpy matrix.
    Reuses logic from Step 1 (Behavioral Extraction).
    """
   
    try:
        # Example logic:
        # df = pd.read_csv(os.path.join(root_path, f"{pid}_aligned.csv"))
        # return df.values
        return np.random.rand(50000, 30) # Placeholder: 30 behavioral features
    except:
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--visual_dir', required=True, help="Path to .npy visual features")
    parser.add_argument('--behavior_dir', required=True, help="Path to OpenFace output")
    parser.add_argument('--output_file', default='data/processed/fused_data.npz')
    args = parser.parse_args()
    
    fuse_streams(args.visual_dir, args.behavior_dir, args.output_file)
