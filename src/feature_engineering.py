import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import joblib

def engineer_features():

    # Load the cleaned labeled dataset
    data_dir = Path('../data')
    cleaned_file = data_dir / 'cleaned_data_with_labels.csv'
    
    df = pd.read_csv(cleaned_file)

    if 'Label' not in df.columns or 'is_attack' not in df.columns:
        print("missing columns")
        return None
    
    # Flow duration statistics
    if 'Flow Duration' in df.columns:
        df['flow_duration_log'] = np.log1p(df['Flow Duration'])
        df['flow_duration_sqrt'] = np.sqrt(df['Flow Duration'])
        print("Created flow duration transformations")
    
    # Byte/packet rate features
    if all(col in df.columns for col in ['Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Flow Duration']):
        df['total_bytes'] = df['Total Length of Fwd Packets'] + df['Total Length of Bwd Packets']
        df['bytes_per_second'] = df['total_bytes'] / (df['Flow Duration'] + 1)  # +1 to avoid divide by zero
        df['bytes_per_second_log'] = np.log1p(df['bytes_per_second'])
        print("Created byte rate features")
    
    if all(col in df.columns for col in ['Total Fwd Packets', 'Total Backward Packets', 'Flow Duration']):
        df['total_packets'] = df['Total Fwd Packets'] + df['Total Backward Packets']
        df['packets_per_second'] = df['total_packets'] / (df['Flow Duration'] + 1)
        df['packets_per_second_log'] = np.log1p(df['packets_per_second'])
        print("Created packet rate features")
    
    # Packet size features
    if all(col in df.columns for col in ['total_bytes', 'total_packets']):
        df['avg_packet_size'] = df['total_bytes'] / (df['total_packets'] + 1)
        df['avg_packet_size_log'] = np.log1p(df['avg_packet_size'])
        print("Created packet size features")
    
    # Forward/Backward ratios
    if all(col in df.columns for col in ['Total Fwd Packets', 'Total Backward Packets']):
        df['fwd_bwd_packet_ratio'] = df['Total Fwd Packets'] / (df['Total Backward Packets'] + 1)
        df['fwd_bwd_packet_ratio_log'] = np.log1p(df['fwd_bwd_packet_ratio'])
        print("Created packet ratio features")
    
    if all(col in df.columns for col in ['Total Length of Fwd Packets', 'Total Length of Bwd Packets']):
        df['fwd_bwd_byte_ratio'] = df['Total Length of Fwd Packets'] / (df['Total Length of Bwd Packets'] + 1)
        df['fwd_bwd_byte_ratio_log'] = np.log1p(df['fwd_bwd_byte_ratio'])
        print("Created byte ratio features")

    #separate into feature and target
    target = df['is_attack']
    df_features = df.drop(['is_attack', 'Label', 'source_file'], axis=1, errors='ignore')
    print(f"Target shape: {target.shape}")
    print(f"Features shape: {df_features.shape}")
    

    numerical_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()

    


    inf_count = np.isinf(df_features[numerical_cols]).sum().sum()
    if inf_count > 0:
        print(f"replacing {inf_count} infinite values with nan")
        df_features[numerical_cols] = df_features[numerical_cols].replace([np.inf, -np.inf], np.nan)
        
        for col in numerical_cols:
            if df_features[col].isnull().sum() > 0:
                median_val = df_features[col].median()
                df_features[col] = df_features[col].fillna(median_val)

    

    scaler = StandardScaler()
    df_scaled = df_features.copy()
    df_scaled[numerical_cols] = scaler.fit_transform(df_features[numerical_cols])
    

    df_scaled['is_attack'] = target
    
    print(f"Shape now: {df_scaled.shape}")
    

    print(f"\nsplitting dataset into train/test sets")
    
    try:
        X = df_scaled.drop('is_attack', axis=1)
        y = df_scaled['is_attack']
        
        # print(f"X shape: {X.shape}")
        # print(f"y shape: {y.shape}")
        # print(f"y unique values: {y.unique()}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # print(f"Training set shape: {X_train.shape}")
        # print(f"Test set shape: {X_test.shape}")
        # print(f"Training set attack rate: {y_train.mean():.2%}")
        # print(f"Test set attack rate: {y_test.mean():.2%}")
        
        train_data = X_train.copy()
        train_data['is_attack'] = y_train
        
        test_data = X_test.copy()
        test_data['is_attack'] = y_test
        
        train_path = data_dir / 'train_data.csv'
        test_path = data_dir / 'test_data.csv'
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        if train_path.exists():
            print(f"train done")
        else:
            print("train fail")
            
        if test_path.exists():
            print(f"test done")
        else:
            print("test fail")
            
    except Exception as e:
        print(f"error: {e}")
        return None

    output_path = data_dir / 'features.csv'

    df_scaled.to_csv(output_path, index=False)

    scaler_path = data_dir / 'scaler.pkl'
    joblib.dump(scaler, scaler_path)
    
    return df_scaled, scaler

def analyze_features():

    data_dir = Path('../data')
    features_file = data_dir / 'features.csv'
    
    df = pd.read_csv(features_file)


    # print(f"\nTarget distribution:")
    # print(df['is_attack'].value_counts())
    # print(f"Attack rate: {df['is_attack'].mean():.2%}")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'is_attack' in numerical_cols:
        numerical_cols.remove('is_attack')
    
    print(f"\nFeature statistics of first 10 features")
    print(df[numerical_cols[:10]].describe())
    

    
    inf_count = np.isinf(df[numerical_cols]).sum().sum()
    if inf_count > 0:
        print(f"found {inf_count} infinite values")
    else:
        print("no infinite values")
    
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"{missing_count} missing values")
    else:
        print("no missing values")

if __name__ == "__main__":
    print("Starting...")
    result = engineer_features()
    
    if result is not None:
        print(f"\n  completed successfully")
        analyze_features()
    else:
        print("failed")
