import os
import json
import argparse
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm


TARGET_SR = 16000
TARGET_DURATION = 8
TARGET_SAMPLES = TARGET_SR * TARGET_DURATION


DEVICE_MAP = {
    'AKGC417L': 0,
    'LittC2SE': 1,
    'Litt3200': 2,
    'Meditron': 3
}

def get_device_id(filename):
    
    parts = filename.split('_')
    dev_name = parts[-1] 
    if dev_name in DEVICE_MAP:
        return DEVICE_MAP[dev_name]
    return -1 

def cyclic_padding(wav, target_len):
  
    curr_len = len(wav)
    if curr_len >= target_len:
        return wav[:target_len] 
    
    
    repeat_count = (target_len // curr_len) + 1
    padded = np.tile(wav, repeat_count)
    return padded[:target_len]

def process_data():
    print("🚀 Begins")

    split_df = pd.read_csv(args.split_file, sep='\t', names=['filename', 'set_type'])
    
    X_train, y_train, device_train = [], [], []
    X_test, y_test, device_test = [], [], []
    
    # İstatistikler
    stats = {'Normal': 0, 'Crackle': 0, 'Wheeze': 0, 'Both': 0}
    
    for index, row in tqdm(split_df.iterrows(), total=split_df.shape[0]):
        fname = row['filename']
        set_type = row['set_type'] 
        
        wav_path = os.path.join(args.data_dir, fname + '.wav')
        txt_path = os.path.join(args.data_dir, fname + '.txt')
        
        if not os.path.exists(wav_path) or not os.path.exists(txt_path):
            continue
            
       
        audio, _ = librosa.load(wav_path, sr=TARGET_SR)
        
       
        dev_id = get_device_id(fname)
        
        
        anns = pd.read_csv(txt_path, sep='\t', names=['start', 'end', 'crackle', 'wheeze'])
        
        for _, ann in anns.iterrows():
            start = int(ann['start'] * TARGET_SR)
            end = int(ann['end'] * TARGET_SR)
            
           
            chunk = audio[start:end]
            
           
            if len(chunk) < 100: 
                continue
                
            # --- CYCLIC PADDING ---
            processed_wav = cyclic_padding(chunk, TARGET_SAMPLES)
            
            #(0:Normal, 1:Crackle, 2:Wheeze, 3:Both)
            c = ann['crackle']
            w = ann['wheeze']
            
            if c == 0 and w == 0:
                label = 0 # Normal
                stats['Normal'] += 1
            elif c == 1 and w == 0:
                label = 1 # Crackle
                stats['Crackle'] += 1
            elif c == 0 and w == 1:
                label = 2 # Wheeze
                stats['Wheeze'] += 1
            else:
                label = 3 # Both
                stats['Both'] += 1
            
            
            if set_type == 'train':
                X_train.append(processed_wav)
                y_train.append(label)
                device_train.append(dev_id)
            else:
                X_test.append(processed_wav)
                y_test.append(label)
                device_test.append(dev_id)

   
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    device_train = np.array(device_train, dtype=np.int64)
    
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int64)
    device_test = np.array(device_test, dtype=np.int64)
    
    
    print(f"Train : {X_train.shape}, Test : {X_test.shape}")
    
    # Kaydet
    np.savez(args.output, 
             X_train=X_train, y_train=y_train, device_train=device_train,
             X_test=X_test, y_test=y_test, device_test=device_test)

    meta = {
        'output': args.output,
        'data_dir': args.data_dir,
        'split_file': args.split_file,
        'target_sr': TARGET_SR,
        'target_duration': TARGET_DURATION,
        'stats': stats
    }
    # Write a small metadata file next to the output
    try:
        with open(args.output + '.meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass

    print(f"✅ Saved: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ICBHI data for AST model")
    parser.add_argument("--data_dir", type=str, default="./data/ICBHI_final_database",
                        help="Path to extracted ICBHI_final_database folder")
    parser.add_argument("--split_file", type=str, default="./data/ICBHI_challenge_train_test.txt",
                        help="Path to the official train/test split file")
    parser.add_argument("--output", type=str, default="icbhi_ast_16k_8s_metadata.npz",
                        help="Output .npz filename")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output file")

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    if not os.path.exists(args.split_file):
        raise FileNotFoundError(f"Split file not found: {args.split_file}")

    if os.path.exists(args.output) and not args.force:
        print(f"Output file {args.output} already exists. Use --force to overwrite.")
    else:
        process_data()
