import pandas as pd 
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_tabular_data(file_path):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported tabular file format")

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    full_text = ""
    
    for page in reader.pages:
        text = page.extract_text()
        full_text += text + "\n"
        
    return full_text

def make_tabular_units(df):
    return df["title"] + " " + df["description"] + " " + df["category"]

def make_pdf_units(text, chunk_size=120):
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

def create_vectors(units):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(units)
    return vectorizer, vectors

def search_similar(query, vectorizer, vectors):
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, vectors)
    return scores

def show_tabular_results(df, scores, top_k=3):
    df = df.copy()
    df["score"] = scores[0]
    results = df.sort_values(by="score", ascending=False).head(top_k)
    print(results[["title", "description", "category", "score"]])

def show_pdf_results(chunks, scores):
    best_index = scores[0].argmax()
    best_score = scores[0][best_index]

    print("Best matching chunk:")
    print(f"Score: {best_score}")
    print(chunks[best_index])
    
def main():
    file_path = input("Enter file path: ")
    query = input("Enter your search query: ")

    if file_path.endswith(".csv") or file_path.endswith(".xlsx"):
        df = load_tabular_data(file_path)
        units = make_tabular_units(df)
        vectorizer, vectors = create_vectors(units)
        scores = search_similar(query, vectorizer, vectors)
        show_tabular_results(df, scores)

    elif file_path.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
        units = make_pdf_units(text)
        vectorizer, vectors = create_vectors(units)
        scores = search_similar(query, vectorizer, vectors)
        show_pdf_results(units, scores)

    else:
        print("Unsupported file format")
    

if __name__ == "__main__":
    main()