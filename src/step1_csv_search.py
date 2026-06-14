import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(file_path):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")
    
    return df

def combine_text(df):
    return df["title"] + " " + df["description"] + " " + df["category"]

def create_vectors(text_data):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(text_data)
    return vectorizer, vectors

def search_similarity(query, vectorizer, vectors):
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, vectors)
    return scores

def show_results(data, scores):
    data["score"] = scores[0]
    sorted_data = data.sort_values(by="score", ascending=False).head(3)
    print(sorted_data[["title", "description", "category", "score"]])

def main():
    data = load_data("data/products.csv")
    combined_text = combine_text(data)
    vectorizer, vectors = create_vectors(combined_text)
    query = input("Enter your search query: ")
    scores = search_similarity(query, vectorizer, vectors)
    show_results(data, scores)
    
if __name__ == "__main__":
    main()