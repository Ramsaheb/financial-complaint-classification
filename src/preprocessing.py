# preprocessing.py — Text cleaning and preprocessing pipeline

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from config import CLEANED_DATA_FILE, PREPROCESSED_DATA_FILE

# Download required NLTK resources (first time only)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

# 1. Load cleaned dataset
df = pd.read_csv(CLEANED_DATA_FILE)

print("Dataset shape before preprocessing:", df.shape)

# 2. Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# 3. Text cleaning function
def preprocess_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()  # lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation/numbers
    tokens = nltk.word_tokenize(text)  # tokenize
    tokens = [t for t in tokens if t not in stop_words]  # remove stopwords
    tokens = [lemmatizer.lemmatize(t) for t in tokens]  # lemmatize
    return " ".join(tokens)

# 4. Apply preprocessing
df['cleaned_text'] = df['complaint_text'].apply(preprocess_text)

# 5. Save preprocessed dataset
df.to_csv(PREPROCESSED_DATA_FILE, index=False)

print("Preprocessing complete.")
print("Dataset shape after preprocessing:", df.shape)
print(df[['complaint_text', 'cleaned_text', 'label']].head())
