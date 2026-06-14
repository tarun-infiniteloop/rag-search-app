from embeddings import load_model
from generation import build_context, build_rag_prompt, ask_llm
from retrieval import search_multiple_sources, print_source
from ingestion import collect_supported_files

def main():
    embedding_model = load_model()
    query = input("Enter your search query: ")
    input_path = input("Enter a file or folder path: ").strip()
    file_paths = collect_supported_files(input_path)
    
    if not file_paths:
        print("No supported files found.")
        return
    
    print("Retrieving...")
    results = search_multiple_sources(file_paths, query, embedding_model, top_k=5)

    print("Building prompt...")
    context = build_context(results)
    prompt = build_rag_prompt(query, context)

    print("Calling Gemini...")
    answer = ask_llm(prompt)

    print("Answer received.")
    print(answer)
    print_source(results)


if __name__ == "__main__":
    main()