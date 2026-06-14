from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    
    full_text = ""
    
    for page in reader.pages:
        text = page.extract_text()
        full_text += text + "\n"
        
    return full_text

def split_into_chunks(text, chunk_size=120):
    words = text.split()
    chunks = []
    current_chunk = ""

    for word in words:
        if len(current_chunk) + len(word) + 1 <= chunk_size:
            current_chunk += word + " "
        else:
            chunks.append(current_chunk.strip())
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks 

def create_vectors(text_chunks):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(text_chunks)
    
    return vectorizer, vectors

def search_similar(query, vectorizer, vectors):
    query_vector =  vectorizer.transform([query])
    scores = cosine_similarity(query_vector, vectors)
    return scores

def show_results(chunks, scores):
    best_index = scores[0].argmax()
    best_score = scores[0][best_index]
    
    print("Best matching chunk:")
    print(f"Score: {best_score}")
    print(chunks[best_index])
        

def main():        
    text = extract_text_from_pdf("data/sample.pdf")
    chunks = split_into_chunks(text)
    vectorizer, vectors = create_vectors(chunks)
    query = input("Enter your PDF search query: ")
    scores = search_similar(query, vectorizer, vectors)
    show_results(chunks, scores)

if __name__ == "__main__":
    main()