import pandas as pd
import numpy as np
from pathlib import Path
import os

def create_labels_from_filename(filename):
    filename_lower = filename.lower()
    
    # attack types based on filename patterns
    if 'morning' in filename_lower and ('ddos' not in filename_lower and 'portscan' not in filename_lower):
        return 'BENIGN'  # Friday-WorkingHours-Morning.pcap_ISCX.csv is benign
    elif 'monday' in filename_lower:
        return 'BENIGN'  # Monday-WorkingHours.pcap_ISCX.csv is benign
    elif 'tuesday' in filename_lower:
        return 'BENIGN'  # Tuesday-WorkingHours.pcap_ISCX.csv is benign
    elif 'wednesday' in filename_lower:
        return 'BENIGN'  # Wednesday-workingHours.pcap_ISCX.csv is benign
    elif 'ddos' in filename_lower:
        return 'DDoS'
    elif 'portscan' in filename_lower:
        return 'PortScan'
    elif 'infilteration' in filename_lower or 'infiltration' in filename_lower:
        return 'Infiltration'
    elif 'webattacks' in filename_lower or 'web-attack' in filename_lower:
        return 'WebAttack'
    elif 'bruteforce' in filename_lower or 'patator' in filename_lower:
        return 'BruteForce'
    elif 'heartbleed' in filename_lower:
        return 'Heartbleed'
    elif 'botnet' in filename_lower:
        return 'Botnet'
    else:
        return 'Unknown'

def merge_and_label_data():
    
    data_dir = Path('../data')
    csv_files = list(data_dir.glob('*.csv'))

    # filter out files that we already processed
    csv_files = [f for f in csv_files if not f.name.startswith(('merged_', 'cleaned_', 'features_', 'train_', 'test_'))]
    
    print(f"Found {len(csv_files)} CSV files to merge")

    dataframes = []

    for i, file in enumerate(csv_files):
        print(f"Loading {file.name}...")
        try:
            df = pd.read_csv(file)
            print(f"Shape: {df.shape}")
            
            #create label based on filename
            label = create_labels_from_filename(file.name)
            df['Label'] = label
            df['source_file'] = file.name
            
            print(f"Assigned label: {label}")
            
            dataframes.append(df)
            
        except Exception as e:
            print(f"Error loading {file.name}: {e}")
    
    if len(dataframes) == 0:
        print("No dataframes were loaded successfully")
        return None

    print(f"Merging {len(dataframes)} dataframes...")
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    # check labels 
    print(f"Label distribution:")
    label_counts = merged_df['Label'].value_counts()
    print(label_counts)
    
    # create binary labels for easier analysis
    merged_df['is_attack'] = (merged_df['Label'] != 'BENIGN').astype(int)
    print(f"Binary label distribution:")
    print(merged_df['is_attack'].value_counts())
    print(f"Attack rate: {merged_df['is_attack'].mean():.2%}")
    
    output_path = data_dir / 'merged_data_with_labels.csv'
    merged_df.to_csv(output_path, index=False)
    print("ok")
    
    return merged_df

if __name__ == "__main__":
    print("Starting merge and label...")
    result = merge_and_label_data()
    
    if result is not None:
        print(f"Merge and label done")
    else:
        print("Merge and label not done") 