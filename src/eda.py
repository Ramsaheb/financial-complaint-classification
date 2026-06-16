# eda.py — Exploratory Data Analysis on the raw complaints dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import RAW_DATA_FILE

# Load dataset
df = pd.read_csv(
    RAW_DATA_FILE,
    encoding="latin1",
    on_bad_lines="skip"
)

# Keep only needed columns
df = df[['Product', 'Consumer complaint narrative']]
df = df.rename(columns={'Consumer complaint narrative': 'complaint_text', 'Product': 'label'})
df = df.dropna()

# Class distribution
print(df['label'].value_counts())

# Bar plot of complaint counts
plt.figure(figsize=(10, 6))
sns.countplot(y=df['label'], order=df['label'].value_counts().index)
plt.title("Distribution of Complaints by Product")
plt.tight_layout()
plt.show()

# Text length distribution
df['text_length'] = df['complaint_text'].apply(len)
plt.figure(figsize=(10, 6))
plt.hist(df['text_length'], bins=50, edgecolor='black')
plt.title("Distribution of Complaint Lengths")
plt.xlabel("Text Length (characters)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
