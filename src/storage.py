import os
import json
import numpy as np
from embeddings import create_embeddings

def save_embeddings(file_name, embeddings):
    np.save(file_name, embeddings)
    
def load_embeddings(file_name):
    return np.load(file_name)

def get_embedding_file_name(file_path):
    base_name = os.path.splitext(file_path)[0]
    return base_name + "_embeddings.npy"

def save_units(file_name, units):
    if hasattr(units, "tolist"):
        units = units.tolist()

    with open(file_name, "w") as f:
        json.dump(units, f)
        
def load_units(file_name):
    with open(file_name, "r") as f:
        return json.load(f)
    
def get_units_file_name(file_path):
    base_name = os.path.splitext(file_path)[0]
    return base_name + "_units.json"

def get_or_create_store(model, units, embedding_file, units_file):
    if os.path.exists(embedding_file) and os.path.exists(units_file):
        embeddings = load_embeddings(embedding_file)
        units = load_units(units_file)
    else:
        embeddings = create_embeddings(model, units)
        save_embeddings(embedding_file, embeddings)
        save_units(units_file, units)

    return embeddings, units