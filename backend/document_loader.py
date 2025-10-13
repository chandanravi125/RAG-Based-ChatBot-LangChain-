"""
This script loads NEC guidelines and Wattmonk documents, preprocesses the text, 
and creates vector stores using the LangChain library.

The vector stores can be used to answer questions based on the NEC guidelines and Wattmonk documents.
"""
import os
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata

# from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
#open_API_key = os.getenv("OPENAI_API_KEY")
google_API_key = os.getenv("GOOGLE_API_KEY")

# Directories
import os

# Get the current working directory
current_dir = os.getcwd()

# Define the data directory
data_dir = os.path.join(current_dir,"data_sources")

# Define the file paths
NEC_PATH = os.path.join(data_dir, "nec_guidelines.pdf")
WATTMONK_PATH = os.path.join(data_dir, "wattmonk.docx")

# Create the output directory
OUT_DIR = os.path.join(current_dir, "chroma_indexes")
os.makedirs(OUT_DIR, exist_ok=True)

def preprocess_text(text):
    """
    Preprocesses the text by converting to lowercase, removing special characters and digits, 
    tokenizing, removing stopwords, and lemmatizing.

    Args:
        text (str): The text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into a string
    text = ' '.join(tokens)
    
    return text

try:
    """
    This loads NEC guidelines and Wattmonk documents, 
    preprocesses the text, and creates vector stores.

    Args:
        loader: The UnstructuredLoader instance.
        emb: The OpenAI embeddings instance of docs

    Returns:
        docs: A list of preprocessed documents.
        Chroma: The vector store instance like vectorstore_nec, vectorstore_watt
    """
        
    
    # Load NEC guidelines
    loader = PyPDFLoader(NEC_PATH)
    docs = loader.load()
    
    # Preprocess text
    preprocessed_docs = [preprocess_text(doc.page_content) for doc in docs]
    
    # Update doc content
    for i, doc in enumerate(docs):
        doc.page_content = preprocessed_docs[i]
    
    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    # Filter complex metadata
    filtered_chunks = filter_complex_metadata(chunks)

    # Embeddings
    emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore_nec = Chroma.from_documents(filtered_chunks, embedding=emb,
    persist_directory=os.path.join(OUT_DIR, "chroma_nec_index"))
    print("💾 Saved NEC index → chroma_nec_index\n")
    
except Exception as e:
    print(f"Error loading NEC guidelines: {str(e)}")


try:
    # Load Wattmonk docs
    loader = UnstructuredLoader(WATTMONK_PATH)
    docs = loader.load()
    
    # Preprocess text
    preprocessed_docs = [preprocess_text(doc.page_content) for doc in docs]
    
    # Update doc content
    for i, doc in enumerate(docs):
        doc.page_content = preprocessed_docs[i]
    
    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    # Filter complex metadata
    filtered_chunks = filter_complex_metadata(chunks)

    # Embeddings
    emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore_watt = Chroma.from_documents(filtered_chunks, embedding=emb,
    persist_directory=os.path.join(OUT_DIR, "chroma_wattmonk_index"))
    
    print("💾 Saved wattmonk index → chroma_wattmonk_index\n")
    print(os.path.join(OUT_DIR, "chroma_wattmonk_index"))

except Exception as e:
    print(f"Error loading Wattmonk docs: {str(e)}")
