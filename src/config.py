# config.py — Project path configuration

from pathlib import Path

# Root directory of the project
ROOT_DIR = Path(__file__).parent.parent

# Data directory
DATA_DIR = ROOT_DIR / "data"

# Dataset file paths
RAW_DATA_FILE = DATA_DIR / "consumercomplaints.csv"
CLEANED_DATA_FILE = DATA_DIR / "cleaned_complaints.csv"
PREPROCESSED_DATA_FILE = DATA_DIR / "preprocessed_complaints.csv"
TRANSLATED_DATA_FILE = DATA_DIR / "translated_complaints.xlsx"