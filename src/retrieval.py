import os
import faiss
import numpy as np 
from ingestion import load_data, make_pdf_chunks_with_pages, combine_text
from storage import get_embedding_file_name, get_units_file_name, get_or_create_store
from embeddings import search_similar_faiss

def build_faiss_index(embeddings):
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index

def build_faiss_retrieval_results(units, indices, distances, metadata=None):
    results = []

    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue

        item = {
            "index": int(idx),
            "distance": float(distances[0][rank]),
            "unit": units[idx]
        }

        if metadata is not None:
            item["metadata"] = metadata[idx]

        results.append(item)

    return results

def save_faiss_index(index, file_name):
    faiss.write_index(index, file_name)

def load_faiss_index(file_name):
    return faiss.read_index(file_name)

def get_faiss_index_file_name(file_path):
    import os
    base_name = os.path.splitext(file_path)[0]
    return base_name + "_faiss.index"

def get_or_create_faiss_index(embeddings, index_file):
    import os

    if os.path.exists(index_file):
        index = load_faiss_index(index_file)
    else:
        index = build_faiss_index(embeddings)
        save_faiss_index(index, index_file)

    return index

def build_tabular_metadata(data, file_path):
    metadata = []

    for _, row in data.iterrows():
        metadata.append({
            "source": "tabular",
            "file_path": file_path,
            "id": int(row["id"]),
            "title": row["title"],
            "category": row["category"]
        })

    return metadata

def build_pdf_metadata(chunks, file_path):
    metadata = []

    for index, chunk in enumerate(chunks):
        metadata.append({
            "source": "pdf",
            "file_path": file_path,
            "chunk_id": index,
            "page": chunk["page"]
        })

    return metadata

def print_source(results):
    print("\nSources:")
    for item in results:
        metadata = item.get("metadata", {})
        source = metadata.get("source")
        file_path = metadata.get("file_path")

        metric_label = ""
        metric_value = None
        if "score" in item:
            metric_label = "score"
            metric_value = item["score"]
        elif "distance" in item:
            metric_label = "distance"
            metric_value = item["distance"]

        if source == "tabular":
            title = metadata.get("title")
            category = metadata.get("category")
            print(
                f"- file: {file_path} | title: {title} | category: {category} | "
                f"{metric_label}: {metric_value}"
            )

        elif source == "pdf":
            chunk_id = metadata.get("chunk_id")
            page = metadata.get("page")
            print(
                f"- file: {file_path} | page: {page} | chunk_id: {chunk_id} | "
                f"{metric_label}: {metric_value}"
            )

        else:
            print(f"- file: {file_path} | {metric_label}: {metric_value}")
            
def search_one_source(file_path, query, model, top_k=3):
    embedding_file = get_embedding_file_name(file_path)
    units_file = get_units_file_name(file_path)
    index_file = get_faiss_index_file_name(file_path)

    if file_path.endswith(".csv") or file_path.endswith(".xlsx"):
        data = load_data(file_path)
        units = combine_text(data)
        embeddings, units = get_or_create_store(model, units, embedding_file, units_file)

        index = get_or_create_faiss_index(embeddings, index_file)
        distances, indices = search_similar_faiss(query, model, index, top_k=top_k)

        metadata = build_tabular_metadata(data, file_path)
        results = build_faiss_retrieval_results(units, indices, distances, metadata=metadata)
        return results

    elif file_path.endswith(".pdf"):
        chunk_objects = make_pdf_chunks_with_pages(file_path)
        units = [chunk["text"] for chunk in chunk_objects]
        embeddings, units = get_or_create_store(model, units, embedding_file, units_file)

        index = get_or_create_faiss_index(embeddings, index_file)
        distances, indices = search_similar_faiss(query, model, index, top_k=top_k)

        metadata = build_pdf_metadata(chunk_objects, file_path)
        results = build_faiss_retrieval_results(units, indices, distances, metadata=metadata)
        return results

    else:
        return []
    
def search_multiple_sources(file_paths, query, model, top_k=3):
    all_results = []

    for file_path in file_paths:
        results = search_one_source(file_path, query, model, top_k=top_k)
        all_results.extend(results)

    all_results.sort(key=lambda item: item["distance"])
    all_results = deduplicate_results(all_results)
    return all_results[:top_k]

def deduplicate_results(results):
    unique_results = []
    seen = set()

    for item in results:
        metadata = item.get("metadata", {})
        unit = item.get("unit")
        key = (metadata.get("source"), metadata.get("title"), unit)

        if key not in seen:
            seen.add(key)
            unique_results.append(item)

    return unique_results