
# Load The Library
import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed

# Load CSV files 
train_df = pd.read_csv(r"C:\Users\sayya\OneDrive\Desktop\train.csv")    # Train.csv file need to load from the device
test_df = pd.read_csv(r"C:\Users\sayya\OneDrive\Desktop\test.csv")      # test.csv file need to load from the device
sample_submission = pd.read_csv(r"C:\Users\sayya\OneDrive\Desktop\sample_submission.csv") # s.csv file need to load from the device

# Clean column names 
train_df.columns = train_df.columns.str.strip().str.lower()
test_df.columns = test_df.columns.str.strip().str.lower()

# Detect actual file column name
file_col = 'file' if 'file' in test_df.columns else test_df.columns[0]

# Paths to audio folders 
train_audio_path = r"C:\Users\sayya\Downloads\shl-intern-hiring-assessment\dataset\audios_train"  # Load the Train_Audio_Path
test_audio_path = r"C:\Users\sayya\Downloads\shl-intern-hiring-assessment\dataset\audios_test"    # Load the Test_Audio_Path

# Feature Extraction Function 
def extract_features(file_path, n_mfcc=13):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        return np.concatenate([mfccs_mean, mfccs_std])
    except Exception as e:
        print(f" Error loading {file_path}: {e}")
        return np.zeros(n_mfcc * 2)

# Extract features from training data (parallelized) 
def process_train_file(row):
    file_name = row["file"]
    score = row["label"]
    features = extract_features(os.path.join(train_audio_path, file_name))
    return features, score

print(" Extracting training features...")
results = Parallel(n_jobs=-1)(delayed(process_train_file)(row) for _, row in tqdm(train_df.iterrows(), total=len(train_df)))
X_train, y_train = zip(*results)
X_train = np.array(X_train)
y_train = np.array(y_train)

#  Train-test split 
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#  Train model 
print(" Training model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_tr, y_tr)

# Evaluate 
y_pred_val = model.predict(X_val)
rmse = sqrt(mean_squared_error(y_val, y_pred_val))
print(f" Validation RMSE: {rmse:.4f}")

# Extract test features (parallelized)
def process_test_file(file_name):
    return extract_features(os.path.join(test_audio_path, file_name))

print(" Extracting test features...")
X_test = Parallel(n_jobs=-1)(
    delayed(process_test_file)(file_name)
    for file_name in tqdm(test_df[file_col])
)
X_test = np.array(X_test)

#Predict on test set 
print(" Predicting on test set...")
test_preds = model.predict(X_test)

# Save submission 
sample_submission["grammar_score"] = test_preds
sample_submission.to_csv("submission.csv", index=False)
print(" Submission saved as 'submission.csv'")
