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
Xtrain.to_csv("tourism_project/data/Xtrain.csv", index=False)
Xtest.to_csv("tourism_project/data/Xtest.csv", index=False)
ytrain.to_csv("tourism_project/data/ytrain.csv", index=False)
ytest.to_csv("tourism_project/data/ytest.csv", index=False)

# Upload files back to Hugging Face dataset repo
files = [
    "tourism_project/data/Xtrain.csv",
    "tourism_project/data/Xtest.csv",
    "tourism_project/data/ytrain.csv",
    "tourism_project/data/ytest.csv"
]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),
        repo_id="rajeshs26/tourism-package-prediction",
        repo_type="dataset",
    )

# Define predictor matrix (X) using selected numeric and categorical features
X = tourism_dataset[numeric_features + categorical_features]

# Define target variable
y = tourism_dataset[target]


# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="rajeshs26/tourism-package-prediction",
        repo_type="dataset",
    )
