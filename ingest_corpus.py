# ingest_corpus.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Define directory paths
CHROMA_PATH = "./chroma_db"
# Place your syllabus PDF in a folder named 'syllabus' or change this path to your file
SYLLABUS_PDF_PATH = "./syllabus/GCSE_computer_science_Syllabus.pdf" 

def main():
    # 1. Ensure the Google Gemini API Key is set in your environment variables
    if "GOOGLE_API_KEY" not in os.environ:
        print("Error: GOOGLE_API_KEY environment variable is not set.")
        print("Please set it in your terminal: export GOOGLE_API_KEY='your-key-here'")
        return

    print(f"--- Step 1: Loading PDF from {SYLLABUS_PDF_PATH} ---")
    if not os.path.exists(SYLLABUS_PDF_PATH):
        print(f"Error: Could not find PDF at {SYLLABUS_PDF_PATH}.")
        print("Please create the directory and place your syllabus PDF inside it.")
        return
        
    loader = PyPDFLoader(SYLLABUS_PDF_PATH)
    raw_documents = loader.load()
    print(f"Successfully loaded {len(raw_documents)} pages.")

    print("\n--- Step 2: Splitting Text into Semantic Chunks ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"Created {len(chunks)} chunks from document pages.")

    print("\n--- Step 3: Generating Embeddings & Building Vector Store ---")
    # Using Google Gemini native embedding model
    #embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    
    # Create persistent Chroma DB instance
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    print(f"--- SUCCESS: Vector database initialized and saved to '{CHROMA_PATH}' ---")

if __name__ == "__main__":
    main()