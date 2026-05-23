# for data manipulation
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi

# Initialize Hugging Face API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Define dataset path (your Hugging Face dataset repo)
DATASET_PATH = "hf://datasets/rajeshs26/tourism-package-prediction/tourism.csv"

# Load dataset
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop unique identifier
df.drop(columns=['CustomerID'], inplace=True)

# Encode categorical features
label_encoders = {}
categorical_cols = ['Gender', 'TypeofContact', 'Occupation', 'MaritalStatus', 'ProductPitched', 'Designation','CityTier']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Define target column
target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload files back to Hugging Face dataset repo
files = [
    "Xtrain.csv",
    "Xtest.csv",
    "ytrain.csv",
    "ytest.csv"
]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),
        repo_id="rajeshs26/tourism-package-prediction",
        repo_type="dataset",
    )

