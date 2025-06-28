import pandas as pd
import numpy as np
from pathlib import Path

def clean_values():

    data_dir = Path('../data')
    input_file = data_dir / 'merged_data_with_labels.csv'

    df = pd.read_csv(input_file)
    print(f"Original shape: {df.shape}")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    inf_counts = np.isinf(df[numerical_cols]).sum()
    total_inf = inf_counts.sum()
    
    if total_inf > 0:
        print(f"Found {total_inf} infinite values across {len(inf_counts[inf_counts > 0])} columns:")
        print(inf_counts[inf_counts > 0])
        for col in numerical_cols:
            if np.isinf(df[col]).any():
                inf_mask = np.isinf(df[col])
                df.loc[inf_mask, col] = np.nan # replace infinite values with NaN
                print(f"Replaced {inf_mask.sum()} infinite values in {col}")
    else:
        print("no infinite values")
    
    
    for col in numerical_cols:      #check for large values
        if df[col].dtype in ['int64', 'float64']:
            max_val = df[col].max()
            if max_val > 1e15:  # large values
                print(f"Column {col} has max value: {max_val}")
                df[col] = df[col].clip(upper=1e15)
    
    
    missing_counts = df.isnull().sum()      # missing values
    total_missing = missing_counts.sum()
    
    if total_missing > 0:
        print(f"Found {total_missing} missing values:")
        print(missing_counts[missing_counts > 0])
        
        #fill missing values with median for numerical columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"Filled missing values in {col} with median: {median_val}")
        
        #fill missing values with mode for categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                print(f"Filled missing values in {col} with mode: {mode_val}")
    else:
        print("no missing values")
    
    # chrck data types and convert
    for col in numerical_cols:
        if df[col].dtype == 'float64':
            #to save memory
            if df[col].min() >= -3.4e38 and df[col].max() <= 3.4e38:
                df[col] = df[col].astype('float32')
                print(f"Converted {col} to float32")

    inf_count = np.isinf(df[numerical_cols]).sum().sum()
    if inf_count > 0:
        print(f"still have infinite values")
    else:
        print("no infinite values remaining")
    
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f" found {missing_count} missing values")
    else:
        print("no missing values found")

    output_path = data_dir / 'cleaned_data_with_labels.csv'
    df.to_csv(output_path, index=False)
    print(f"done")

    
    return df

if __name__ == "__main__":
    print("starting...")
    result = clean_values()
    
    if result is not None:
        print(f"\nok!")
    else:
        print("cleaning failed") 