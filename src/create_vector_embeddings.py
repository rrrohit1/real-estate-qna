from transformers import BertTokenizer, BertModel
import torch

def create_embeddings(chunks):
    """
    Create vector embeddings for a list of text chunks using BERT model.

    Args:
        chunks (list): A list of text chunks.

    Returns:
        embeddings (list): A list of vector embeddings for each text chunk.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    embeddings = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    
    return embeddings

# Example usage
# embeddings = create_embeddings(chunks)