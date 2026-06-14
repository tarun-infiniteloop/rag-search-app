import os
import pandas as pd
from pypdf import PdfReader

def load_data(file_path):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")
    
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        full_text += text + "\n"

    return full_text

def make_pdf_chunks(text, chunk_size=120):
    words = text.split()
    chunks = []
    current_chunk = ""

    for word in words:
        if len(current_chunk) + len(word) + 1 <= chunk_size:
            current_chunk += word + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = word + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def combine_text(df):
    return df["title"] + " " + df["description"] + " " + df["category"]

def make_pdf_chunks_with_pages(file_path, chunk_size=120):
    reader = PdfReader(file_path)
    chunks = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text:
            continue

        words = text.split()
        current_chunk = ""

        for word in words:
            if len(current_chunk) + len(word) + 1 <= chunk_size:
                current_chunk += word + " "
            else:
                chunks.append({
                    "page": page_num,
                    "text": current_chunk.strip()
                })
                current_chunk = word + " "

        if current_chunk:
            chunks.append({
                "page": page_num,
                "text": current_chunk.strip()
            })

    return chunks

def collect_supported_files(path):
    supported_extensions = (".csv", ".xlsx", ".pdf")

    if os.path.isfile(path):
        if path.endswith(supported_extensions):
            return [path]
        return []

    if os.path.isdir(path):
        collected_files = []

        for root, _, files in os.walk(path):
            for file_name in files:
                full_path = os.path.join(root, file_name)
                if full_path.endswith(supported_extensions):
                    collected_files.append(full_path)

        return sorted(collected_files)

    return []
    