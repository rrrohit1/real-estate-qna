import os
import pandas as pd
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
# from transformers import AutoTokenizer, AutoModel
# import torch
import PyPDF2
import textract

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(file_path):
    text = textract.process(file_path).decode('utf-8')
    return text

def extract_text_from_xlsx(file_path):
    df = pd.read_excel(file_path)
    text = df.astype(str).values.flatten()
    return "\n".join(text)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Define paths to the dataset
data_dir = '/Users/rohitrawat/github-repos/real-estate-qna/data/Tembusu Grand'
text_data = []

for root, dirs, files in os.walk(data_dir):
    for file in files:
        file_path = os.path.join(root, file)

        if file.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif file.endswith('.docx'):
            text = extract_text_from_docx(file_path)
        elif file.endswith('.xlsx'):
            text = extract_text_from_xlsx(file_path)
        else:
            continue
        text = clean_text(text)
        text = f"Document Name: {file_path}\n\n" + text
        text_data.append(text+"\n\n\n")

# Combine all text data
corpus = " ".join(text_data)
# Save corpus to a txt file
with open('/Users/rohitrawat/github-repos/real-estate-qna/data/corpus.txt', 'w') as file:
    file.write(corpus)
print(corpus[:1000])
