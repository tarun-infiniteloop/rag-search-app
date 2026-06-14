from sentence_transformers import SentenceTransformer
import numpy as np

def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

def create_embeddings(model, text_data):
    if hasattr(text_data, "tolist"):
        text_data = text_data.tolist()
    embeddings = model.encode(text_data)
    return embeddings

def search_similar_faiss(query, model, index, top_k=3):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)
    return distances, indices