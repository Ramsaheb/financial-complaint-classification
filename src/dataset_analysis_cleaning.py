# dataset_analysis_cleaning.py — Load, analyze, and clean the raw complaints dataset

import pandas as pd
from config import RAW_DATA_FILE, CLEANED_DATA_FILE

# 1. Load dataset
df = pd.read_csv(
    RAW_DATA_FILE,
    encoding="latin1",
    on_bad_lines="skip"
)

# 2. View dataset shape
print("Dataset shape:", df.shape)

# 3. Display first 5 rows
print("\nSample rows:")
print(df.head())

# 4. Dataset info (column types, null counts)
print("\nDataset Info:")
print(df.info())

# 5. Check missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# 6. Unique value counts for categorical columns
print("\nUnique values per column:")
for col in ["Product", "Sub-product", "Issue", "Sub-issue"]:
    if col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values")

# 7. Distribution of target variable (Product)
print("\nComplaint distribution by Product:")
print(df["Product"].value_counts())

# 8. Drop irrelevant columns
df_clean = df.drop(columns=[
    'Unnamed: 0',
    'Date received',
    'Sub-product',
    'Issue',
    'Sub-issue'
])

# 9. Rename for clarity
df_clean = df_clean.rename(columns={
    'Consumer complaint narrative': 'complaint_text',
    'Product': 'label'
})

# 10. Drop rows with missing values in key columns
df_clean = df_clean.dropna(subset=['complaint_text', 'label'])

print("Cleaned dataset shape:", df_clean.shape)
print(df_clean)

# 11. Save cleaned dataset
df_clean.to_csv(CLEANED_DATA_FILE, index=False)
