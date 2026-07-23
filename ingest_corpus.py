import os
import tomllib  # Built-in in Python 3.11+

# --- Load Google API Key from Streamlit secrets.toml ---
secrets_path = os.path.join(".streamlit", "secrets.toml")
if os.path.exists(secrets_path):
    with open(secrets_path, "rb") as f:
        secrets = tomllib.load(f)
        api_key = secrets.get("GOOGLE_API_KEY") or secrets.get("GEMINI_API_KEY")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            os.environ["GEMINI_API_KEY"] = api_key

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

CHROMA_PATH = "./chroma_db"
DATA_PATH = "./syllabus"

TOPIC_MAP = {
    "3.3.1": "AQA 3.3.1: Number Bases & Conversions (Binary, Hex)",
    "3.3.2": "AQA 3.3.2: Units of Information & Binary Arithmetic",
    "3.3.3": "AQA 3.3.3: Character Encoding (ASCII & Unicode)",
    "3.3.4": "AQA 3.3.4: Representing Images & Sound",
    "3.3.5": "AQA 3.3.5: Data Compression (Huffman & RLE)",
    "3.4.1": "AQA 3.4.1: Hardware & Software Classification",
    "3.4.2": "AQA 3.4.2: Systems Architecture (CPU & Von Neumann)",
    "3.4.3": "AQA 3.4.3: Memory (RAM, ROM & Cache)",
    "3.4.4": "AQA 3.4.4: Secondary Storage Devices",
    "3.5.1": "AQA 3.5.1: Network Types & Connections (PAN, LAN, WAN)",
    "3.5.2": "AQA 3.5.2: Network Topologies & Data Routing (Star, Bus, Packets)",
    "3.5.3": "AQA 3.5.3: Protocols & The 4-Layer TCP/IP Model",
    "3.6.1": "AQA 3.6.1: Cyber Security Threats & Attacks",
    "3.6.2": "AQA 3.6.2: Social Engineering & Malware Methods",
    "3.6.3": "AQA 3.6.3: Cyber Security Prevention & Detection",
    "3.7.1": "AQA 3.7.1: Relational Database Concepts & Structure",
    "3.7.2": "AQA 3.7.2: Primary Keys, Foreign Keys & Relationships",
    "3.8.1": "AQA 3.8.1: Ethical & Privacy Issues",
    "3.8.2": "AQA 3.8.2: Environmental Impact of Technology",
    "3.8.3": "AQA 3.8.3: Computer Legislation (DPA, CMA, Copyright)",
}

def detect_subtopic(text: str, filename: str) -> str:
    combined_str = f"{filename} {text[:300]}"
    for code, full_label in TOPIC_MAP.items():
        if code in combined_str:
            return full_label
    return "AQA 3.5.1: Network Types & Connections (PAN, LAN, WAN)"

def build_vector_db():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created directory {DATA_PATH}.")
        return

    print("Loading PDF syllabus files...")
    # Configured specifically to load .pdf files
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    raw_docs = loader.load()

    if not raw_docs:
        print(f"No PDF documents found in {DATA_PATH}. Ingestion aborted.")
        return

    print(f"Loaded {len(raw_docs)} page(s). Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(raw_docs)

    print("Assigning subtopic metadata tags...")
    for chunk in chunks:
        source_file = chunk.metadata.get("source", "")
        detected = detect_subtopic(chunk.page_content, source_file)
        chunk.metadata["sub_topic"] = detected

    print("Generating embeddings and indexing to ChromaDB...")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview",google_api_key=os.getenv("GOOGLE_API_KEY"))
    
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    print(f"✅ Ingestion complete! Persisted {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    build_vector_db()