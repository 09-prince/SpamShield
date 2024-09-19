import nltk
import os

# Define the directory where NLTK data will be downloaded
nltk_data_dir = '/opt/render/project/src/.venv/nltk_data'
os.makedirs(nltk_data_dir, exist_ok=True)

# Download the stopwords data to the specified directory
nltk.download('stopwords', download_dir=nltk_data_dir)
