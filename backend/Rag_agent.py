import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from intent_router import detect_intent

# Load env vars
load_dotenv()

class RAGAgent:
    """ Retrieval-Augmented Generator (RAG) agent using Gemini. """
    def __init__(self, nec_index, watt_index, llm):
        self.nec_index = nec_index
        self.watt_index = watt_index
        self.llm = llm

    def answer(self, query):
        """ Answers a query based on intent and vector context. """
        intent = detect_intent(query)
        print(f"Intent: {intent}")

        if intent == "nec":
            docs = self.nec_index.similarity_search(query, k=3)
        elif intent == "wattmonk":
            docs = self.watt_index.similarity_search(query, k=3)
        else:
            docs = []

        context = "\n\n".join([d.page_content for d in docs])
        final_prompt = f"Answer the user query based on the following context (if any):\n{context}\n\nQuery: {query}"
        response = self.llm.invoke(final_prompt)
        return response.content

def create_rag_agent():
    """ Creates and configures the Gemini-based RAG Agent. """
    google_key = os.getenv("GOOGLE_API_KEY")
    if not google_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")

    emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=google_key)

    vector_nec = Chroma(persist_directory="chroma_indexes/chroma_nec_index", embedding_function=emb)
    vector_watt = Chroma(persist_directory="chroma_indexes/chroma_wattmonk_index", embedding_function=emb)

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_key, temperature=0.2)

    return RAGAgent(vector_nec, vector_watt, llm)

if __name__ == "__main__":
    agent = create_rag_agent()
    query = "What are NEC guidelines for electrical safety?"
    print(agent.answer(query))
