import os
import requests
from flask import Flask, request, render_template
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Elasticsearch client
es_client = Elasticsearch(
    "https://34.87.88.152:9243",
    api_key=os.getenv("ES_API_KEY"),
    verify_certs=False,
)

index_source_fields = {
    "general-rules": ["content"]
}

def get_elasticsearch_results(query):
    es_query = {
        "query": {
            "semantic": {
                "field": "content.semantic",
                "query": query
            }
        },
        "highlight": {
            "fields": {
                "content.semantic": {
                    "order": "score",
                    "number_of_fragments": 1
                }
            }
        }
    }
    result = es_client.search(index="general-rules", body=es_query)
    return result["hits"]["hits"]

def create_openai_prompt(results):
    context = ""
    for i, hit in enumerate(results):
        if "highlight" in hit:
            highlighted_texts = []
            for values in hit["highlight"].values():
                highlighted_texts.extend(values)
            context += "\n --- \n".join(highlighted_texts)
        else:
            source_field = index_source_fields.get(hit["_index"])[0]
            hit_context = hit["_source"][source_field]
            context += f"[{i+1}] {hit_context}\n"
    prompt = f"""
Instructions:

- You are an assistant for question-answering tasks.
- Answer questions truthfully and factually using only the context presented.
- If you don't know the answer, just say that you don't know, don't make up an answer.
- Use markdown format for code examples.
- You are correct, factual, precise, and reliable.

Context:
{context}
"""
    return prompt

def generate_local_completion(user_prompt, question):
    url = "http://localhost:1234/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "model": "local-model",  # Optional; or remove it if LM Studio doesn't require it
        "messages": [
            {"role": "system", "content": user_prompt},
            {"role": "user", "content": question}
        ],
        "temperature": 0.7,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error calling local model: {str(e)}"

@app.route("/", methods=["GET", "POST"])
def index():
    response = None
    question = None
    if request.method == "POST":
        question = request.form["question"]
        results = get_elasticsearch_results(question)
        context_prompt = create_openai_prompt(results)
        response = generate_local_completion(context_prompt, question)
    return render_template("index.html", response=response, question=question)

if __name__ == "__main__":
    app.run(debug=True)
