from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import requests

# -------- ENDEE CONFIG -------- #
ENDEE_URL = "http://localhost:8080/api/v1"
COLLECTION_NAME = "pdf_data"

# -------- LOAD MODEL -------- #
MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

print("✅ Model + Endee Ready!")

# -------- CREATE COLLECTION -------- #
def create_collection():
    try:
        requests.post(f"{ENDEE_URL}/collections", json={
            "name": COLLECTION_NAME,
            "dimension": 768
        })
    except:
        pass  # already exists

# -------- PROCESS PDF -------- #
def process_pdf(file):
    create_collection()

    pdf = PdfReader(file)
    text = ""

    for page in pdf.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"

    if not text.strip():
        return "❌ No text found in PDF"

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    # 🔥 STORE IN ENDEE
    for i, chunk in enumerate(chunks):
        vector = embeddings.embed_query(chunk)

        requests.post(
            f"{ENDEE_URL}/collections/{COLLECTION_NAME}/vectors",
            json={
                "id": str(i),
                "values": vector,
                "metadata": {"text": chunk}
            }
        )

    return f"✅ Stored {len(chunks)} chunks in Endee"

# -------- ASK QUESTION -------- #
def ask_question(query: str):

    # 🔥 EMBED QUERY
    query_vector = embeddings.embed_query(query)

    # 🔥 SEARCH ENDEE
    res = requests.post(
        f"{ENDEE_URL}/collections/{COLLECTION_NAME}/search",
        json={
            "vector": query_vector,
            "top_k": 5
        }
    )

    results = res.json().get("results", [])

    if not results:
        return "No relevant information found."

    context = "\n".join([r["metadata"]["text"] for r in results])

    # 🔥 PROMPT
    prompt = f"""
Answer the question using the context.

Give:
- Definition
- Explanation
- One example

Keep answer clear in 3-5 sentences.

Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    print("⏳ Generating answer...")

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    return answer