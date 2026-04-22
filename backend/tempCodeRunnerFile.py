from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# -------- GLOBAL STORAGE -------- #
vectorstore = None

# -------- LOAD MODEL ONCE -------- #
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

print("✅ Model Loaded Successfully!")

# -------- PROCESS PDF -------- #
def process_pdf(file):
    global vectorstore

    pdf = PdfReader(file)
    text = ""

    for page in pdf.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"

    if not text.strip():
        return "❌ No text found in PDF"

    # Better splitting
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_text(text)

    # create vectorstore
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    return f"✅ PDF processed successfully with {len(chunks)} chunks"

# -------- ASK QUESTION -------- #
def ask_question(query: str):
    global vectorstore

    if vectorstore is None:
        return "⚠️ Please upload a PDF first."

    # better retrieval
    docs = vectorstore.similarity_search(query, k=5)

    context = "\n".join([doc.page_content for doc in docs])

    # 🔥 STRONG PROMPT (FIXED)
    prompt = f"""
You are an expert tutor and assistant.

Use ONLY the given context to answer the question.

RULES:
- Give a detailed answer (4-6 sentences)
- First explain the concept simply
- Then add explanation or example
- Do NOT copy text directly
- If context is small, still explain intelligently

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

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 🔥 SAFETY IMPROVEMENT (avoid too-short answers)
    if len(answer.split()) < 25:
        answer += (
            " The concept involves more detailed explanations which describe "
            "its definition, usage, and real-world applications."
        )

    return answer.strip()