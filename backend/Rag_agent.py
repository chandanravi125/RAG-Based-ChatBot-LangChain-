
""" 
This script sets up a Retrieval-Augmented Generator (RAG) agent using LangChain.
It loads pre-existing vector stores, creates retrievers, and initializes a ChatOpenAI model.
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from intent_router import detect_intent

# Load environment variables from .env file
load_dotenv()

class RAGAgent:
    """ 
    A Retrieval-Augmented Generator (RAG) agent.
    """
    def __init__(self, nec_index, watt_index, llm):
        """ 
        Initializes the RAG agent.

        Args:
            nec_index: The NEC index.
            watt_index: The Wattmonk index.
            llm: The language model.
        """
        self.nec_index = nec_index
        self.watt_index = watt_index
        self.llm = llm

    def answer(self, query):
        """ 
        Answers a query based on the intent.

        Args:
            query: The query to answer.

        Returns:
            The response to the query.
        """
        intent = detect_intent(query)
        print(f" Intent: {intent}")

        if intent == "nec":
            docs = self.nec_index.similarity_search(query, k=3)
        elif intent == "wattmonk":
            docs = self.watt_index.similarity_search(query, k=3)
        else:
            docs = []

        context = "\n\n".join([d.page_content for d in docs])
        final_prompt = f"Answer the user query based on the following context (if any):\n{context}\n\nQuery: {query}"
        resp = self.llm(final_prompt)
        return resp

def create_rag_agent(open_API_key):
    """ 
    Creates a RAG agent.

    Args:
        open_API_key: The OpenAI API key.

    Returns:
        A RAGAgent instance.
    """
    emb = OpenAIEmbeddings(api_key=open_API_key)
    vector_Store_nec = Chroma.load_local("chroma_nec_index", emb)
    vector_Store_watt = Chroma.load_local("chroma_wattmonk_index", emb)
    llm = ChatOpenAI(openai_api_key=open_API_key, temperature=0.0, model="gpt-4o")
    return RAGAgent(vector_Store_nec, vector_Store_watt, llm)

def main():
    try:
        # Set OpenAI API key
        open_API_key = os.getenv("OPENAI_API_KEY")
        if not open_API_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        rag_agent = create_rag_agent(open_API_key)

        # Test the RAG agent
        query = "What are the NEC guidelines for electrical safety?"
        response = rag_agent.answer(query)
        print(response)

    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
