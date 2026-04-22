📄 PDF Chatbot using RAG + Endee
👋 About the Project

This project is a simple but practical implementation of a PDF Question Answering system.

The idea is straightforward:
Instead of manually reading long PDFs, users can upload a document and directly ask questions. The system understands the content, finds relevant information, and generates answers in natural language.

While building this, the main focus was not just on getting answers, but on understanding how modern AI systems actually work — especially RAG (Retrieval-Augmented Generation) and vector databases like Endee.

⚙️ How the System Works (Simple Explanation)
A user uploads a PDF
The system reads and extracts text
The text is broken into smaller chunks
Each chunk is converted into embeddings (numerical vectors)
These vectors are stored in Endee Vector Database
When a question is asked:
The question is also converted into a vector
Endee finds the most relevant chunks
The model (FLAN-T5) uses this context to generate an answer

🧠 Architecture Overview
User → FastAPI Backend  
     → PDF Processing  
     → Text Chunking  
     → Embeddings  
     → Endee Vector Database  
     → Semantic Search  
     → Language Model (FLAN-T5)  
     → Final Answer

🛠️ Tech Stack
Python
FastAPI (Backend API)
Endee (Vector Database)
Transformers (HuggingFace) – FLAN-T5 model
Sentence Transformers – for embeddings
PyPDF2 – for PDF text extraction
Docker – for running Endee server

⚠️ Limitations
The model used (FLAN-T5-base) is lightweight, so answers may not always be perfect
Performance depends on the quality of the PDF text
No frontend UI yet (currently API-based interaction)

🔮 Future Improvements
Add a simple chat UI
Support multiple PDFs
Use a stronger LLM for better answers
Show source chunks along with answers
Add streaming responses

📌 Key Learning Outcomes

While building this project, I learned:

How RAG pipelines work in real systems
Why vector databases are important in AI applications
How to connect LLMs with external data sources
Practical challenges like latency, prompt design, and retrieval quality

📝 Final Note

This project is a step toward building real-world AI systems rather than just using pre-built tools.
The focus was on understanding the pipeline, integrating components, and creating something functional and meaningful.

Kartikey Pathak
B.Tech CSE Uttarachal University(2027)

Screenshots:-

API Interface
<img width="1920" height="1080" alt="Screenshot (136)" src="https://github.com/user-attachments/assets/63761eb6-1461-4d7f-957c-2f450f13ce5e" /> 

Sample Response
<img width="1920" height="1080" alt="Screenshot (137)" src="https://github.com/user-attachments/assets/f864d18a-1f32-40f2-bc0e-e3b0e9f8ea6f" />
