#  Gemini RAG Chatbot — NEC & Wattmonk Assistant

This project is a **Retrieval-Augmented Generation (RAG)** chatbot built using **LangChain** and **Gemini (Google Generative AI)**.  
It can answer queries related to **NEC (National Electrical Code)** and **Wattmonk company documents**, with intelligent intent classification and contextual retrieval.

---

##  Features

- Hybrid intent classification (`NEC`, `WATTMONK`, `GENERAL`)
- RAG pipeline powered by **Gemini embeddings + Gemini chat models**
-  Uses **LangChain + Chroma** for document retrieval
-  REST API using **Flask**
-  Simple UI using **Streamlit**
-  Modular structure for easy updates and debugging

---

##  Project Structure

project/
│
├── data_sources/
│ ├── nec_guidelines.pdf
│ └── wattmonk.docx
│
├── chroma_indexes/
│ ├── chroma_nec_index/
│ └── chroma_wattmonk_index/
│
├── document_loader.py # Builds Chroma indexes from documents
├── Rag_agent.py # RAG logic + Gemini LLM integration
├── intent_router.py # Intent detection (NEC/Wattmonk/General)
├── main.py # Flask API backend (/ask)
├── streamlit.py # Streamlit frontend
├── requirements.txt
└── README.md



---

## 🧠 How It Works

1. `document_loader.py`
   Loads and preprocesses NEC & Wattmonk documents → creates vector stores with **Gemini embeddings**

2. `internal_routing.py` 
   Classifies user intent → decides which vector store to use (NEC or Wattmonk).

3. `rag_agent.py`
   Uses **retrievers + Gemini LLM** to generate accurate, context-aware answers.

4. `main.py`
   Flask API that accepts `POST` requests at `/ask`.

5. `app.py`
   Streamlit web UI for chatting with the assistant.

---

## Setup Instructions


2️⃣ Create a Virtual Environment

python -m venv venv

venv\Scripts\activate       

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Add Environment Variable

Create a .env file in the project root:

GOOGLE_API_KEY=your_gemini_api_key_here


# Building Vector Indexes

python document_loader.py
This will generate:
/chroma_indexes/chroma_nec_index/
/chroma_indexes/chroma_wattmonk_index/

# Running the Chatbot

Start the Flask API:
python main.py
API runs at → http://127.0.0.1:5000/ask

Start the Streamlit UI:
streamlit run app.py
UI runs at → http://localhost:8501

🗨️ Example Usage
POST request:
curl -X POST http://127.0.0.1:5000/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "What is Article 310 in NEC?"}'

Response:
{
  "response": "Article 310 in NEC covers the general requirements for conductors, including insulation, ampacity, and installation conditions."
}

# Tech Stack
Component	Technology
Embeddings	Gemini Embeddings (models/gemini-embedding-001)
LLM	Gemini 1.5 Pro
Framework	LangChain
Database	Chroma Vector Store
Backend	Flask
Frontend	Streamlit
Language	Python 3.10+

# Optional: Rebuild Index Automatically
If your document sources are updated, simply rerun:

python document_loader.py
