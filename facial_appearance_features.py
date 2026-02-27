import pandas as pd
import numpy as np
import os
import re
import glob
from tqdm import tqdm

WINDOW_LENGTH_FRAMES = 5400 

# param
OVERLAP_RATIO = 0.90 

# Path Configuration
ROOT_DATA_PATH = 'data/raw/' 
OUTPUT_PATH = 'data/processed/'

AU_R_FEATNAMES = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU09_r', 
                  'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 
                  'AU25_r', 'AU26_r'] # Intensity
AU_C_FEATNAMES = ['AU04_c', 'AU12_c', 'AU15_c', 'AU23_c', 'AU28_c', 'AU45_c'] # Presence
POSE_FEATNAMES = ['pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz']
GAZE_FEATNAMES = ['gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z']

def generate_sliding_windows(data, win_len, overlap_ratio):
    """
    Implements the SlidingWindow function from Algorithm 1.
    Splits (Frames x Features) into (N_windows x win_len x Features).
    
    Args:
        data (np.array): Input feature matrix (Frames, Features)
        win_len (int): Length of window in frames
        overlap_ratio (float): Percentage of overlap (0.0 to 1.0)
        
    Returns:
        np.array: Windowed data of shape (N_windows, win_len, Features)
    """
    n_frames, n_features = data.shape
    
    if n_frames < win_len:
        return None

    step_size = int(win_len * (1 - overlap_ratio))
    if step_size < 1: step_size = 1

    windows = []
    
    # Sliding window logic
    for start_idx in range(0, n_frames - win_len + 1, step_size):
        end_idx = start_idx + win_len
        window = data[start_idx:end_idx, :]
        windows.append(window)
        
    if not windows:
        return None
        
    return np.array(windows)

def load_metadata(root_path):
    """
    Loads and combines metadata files for AVEC datasets (Train/Dev/Test).
    """
    metadata_files = glob.glob(os.path.join(root_path, '*split*.csv'))
    dfs = []
    
    for f in metadata_files:
        try:
            df = pd.read_csv(f)
            # Determine split from filename if possible
            if 'train' in f.lower(): df['split'] = 'train'
            elif 'dev' in f.lower(): df['split'] = 'dev'
            elif 'test' in f.lower(): df['split'] = 'test'
            else: df['split'] = 'unknown'
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not dfs:
        raise FileNotFoundError("No metadata CSVs found in root path.")
        
    full_meta = pd.concat(dfs, ignore_index=True)
    
    if 'Participant_ID' in full_meta.columns:
        full_meta.set_index('Participant_ID', inplace=True)
    return full_meta

def process_participant(pid, folder_path, metadata_row):
    """
    Extracts AUs, Gaze, and Pose, merges them, and applies windowing.
    """
    try:
        # 1. Load OpenFace Output Files
        f_pose = os.path.join(folder_path, f"{pid}_CLNF_pose.txt")
        f_gaze = os.path.join(folder_path, f"{pid}_CLNF_gaze.txt")
        f_au = os.path.join(folder_path, f"{pid}_CLNF_AUs.txt")

        # Check existence
        if not (os.path.exists(f_pose) and os.path.exists(f_gaze) and os.path.exists(f_au)):
            return None

        # Read CSVs
        df_pose = pd.read_csv(f_pose, sep=',', skipinitialspace=True)
        df_gaze = pd.read_csv(f_gaze, sep=',', skipinitialspace=True)
        df_au = pd.read_csv(f_au, sep=',', skipinitialspace=True)
        
        # Cleanup Columns
        for df in [df_pose, df_gaze, df_au]:
            df.columns = df.columns.str.strip()

        # Rename for consistency
        df_pose.rename(columns={'Tx': 'pose_Tx', 'Ty': 'pose_Ty', 'Tz': 'pose_Tz', 
                                'Rx': 'pose_Rx', 'Ry': 'pose_Ry', 'Rz': 'pose_Rz', 
                                'timestamp': 'timestamp'}, inplace=True)
        if 'x_0' in df_gaze.columns:
            df_gaze.rename(columns={'x_0': 'gaze_0_x', 'y_0': 'gaze_0_y', 'z_0': 'gaze_0_z', 
                                    'x_1': 'gaze_1_x', 'y_1': 'gaze_1_y', 'z_1': 'gaze_1_z',
                                    'timestamp': 'timestamp'}, inplace=True)
        
        df_au.rename(columns={'timestamp': 'timestamp'}, inplace=True)

        # 2. Merge Modalities based on Timestamp
        df_merged = pd.merge(df_au, df_gaze, on='timestamp', how='inner')
        df_merged = pd.merge(df_merged, df_pose, on='timestamp', how='inner')
        
        df_merged.sort_values('timestamp', inplace=True)
        
        # 3. Extract Feature Subsets
        feats_au_r = df_merged[[c for c in AU_R_FEATNAMES if c in df_merged.columns]].values
        feats_pose = df_merged[[c for c in POSE_FEATNAMES if c in df_merged.columns]].values
        feats_gaze = df_merged[[c for c in GAZE_FEATNAMES if c in df_merged.columns]].values
        
        X_behavior = np.hstack([feats_au_r, feats_pose, feats_gaze])
        
        # 4. Apply Multi-Time Scale Windowing
        windows = generate_sliding_windows(X_behavior, WINDOW_LENGTH_FRAMES, OVERLAP_RATIO)
        
        if windows is None:
            return None
            
        return {
            'pid': pid,
            'windows': windows, 
            'label': metadata_row['PHQ8_Score'],
            'split': metadata_row['split']
        }

    except Exception as e:
        print(f"Error processing {pid}: {e}")
        return None

def main():
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        
    print("--- Loading Metadata ---")
    metadata_df = load_metadata(ROOT_DATA_PATH)
    print(f"Metadata loaded for {len(metadata_df)} participants.")

    # Locate participant folders
    participant_folders = [f for f in os.listdir(ROOT_DATA_PATH) if f.endswith('_P')]
    
    processed_data = []

    print(f"--- Starting Extraction (Window: {WINDOW_LENGTH_FRAMES} frames, Overlap: {int(OVERLAP_RATIO*100)}%) ---")
    
    for p_folder in tqdm(participant_folders):
        try:
            pid_str = re.match(r'(\d+)_P', p_folder).group(1)
            pid = int(pid_str)
            
            if pid not in metadata_df.index:
                continue
                
            full_folder_path = os.path.join(ROOT_DATA_PATH, p_folder)
            
            result = process_participant(pid, full_folder_path, metadata_df.loc[pid])
            
            if result:
                processed_data.append(result)
                
        except AttributeError:
            continue

    # Save Results
    print(f"--- Saving {len(processed_data)} processed participants to {OUTPUT_PATH} ---")
    
    save_file = os.path.join(OUTPUT_PATH, 'msfe_processed_data.pkl')
    pd.to_pickle(processed_data, save_file)
    print("Done.")

if __name__ == "__main__":
    main()
