from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from .model import process_pdf, ask_question

app = FastAPI()

# -------- Upload PDF -------- #
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    result = process_pdf(file.file)
    return {"message": result}

# -------- Ask Question -------- #
class Query(BaseModel):
    question: str

@app.post("/ask")
def get_answer(query: Query):
    answer = ask_question(query.question)
    return {"answer": answer}