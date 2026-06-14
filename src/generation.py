import os
from google import genai
from dotenv import load_dotenv

load_dotenv()


def build_context(results):
    context_parts = []

    for item in results:
        metric_line = ""
        if "score" in item:
            metric_line = f"Score: {item['score']}\n"
        elif "distance" in item:
            metric_line = f"Distance: {item['distance']}\n"

        context_parts.append(
            f"Source: {item['metadata']['source']}\n"
            f"File: {item['metadata']['file_path']}\n"
            f"{metric_line}"
            f"Content: {item['unit']}\n"
        )

    return "\n---\n".join(context_parts)


def build_rag_prompt(query, context):
    prompt = (
        "You are a helpful assistant. Answer the user's question using only the provided context.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )
    return prompt


def ask_llm(prompt):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )
    return response.text