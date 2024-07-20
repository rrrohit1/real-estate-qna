import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

nltk.download('punkt')

def semantic_similarity_chunking(text, max_chunk_size=500):
    sentences = nltk.sent_tokenize(text)
    
    # Load pre-trained BERT model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    # Generate sentence embeddings
    embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(embeddings)
    
    # Create chunks based on similarity
    chunks = []
    current_chunk = [sentences[0]]
    for i in range(1, len(sentences)):
        if similarity_matrix[i-1][i] > 0.7 and len(' '.join(current_chunk)) + len(sentences[i]) <= max_chunk_size:
            current_chunk.append(sentences[i])
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentences[i]]
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# Example usage
# with open('data/corpus.txt', 'r') as file:
#     text = file.read()

# chunks = semantic_similarity_chunking(text)