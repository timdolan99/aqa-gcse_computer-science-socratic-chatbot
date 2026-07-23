import os
import tomllib
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- Automatically load API key from .streamlit/secrets.toml ---
secrets_path = os.path.join(".streamlit", "secrets.toml")
if os.path.exists(secrets_path):
    with open(secrets_path, "rb") as f:
        secrets = tomllib.load(f)
        api_key = secrets.get("GOOGLE_API_KEY") or secrets.get("GEMINI_API_KEY")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            os.environ["GEMINI_API_KEY"] = api_key

CHROMA_PATH = "./chroma_db"
DATA_PATH = "./syllabus"

TOPIC_MAP = {
    "10": "Edexcel Topic 10: Medicine in Britain (c1250–present) & Western Front",
    "11": "Edexcel Topic 11: Crime and Punishment in Britain (c1000–present)",
    "12": "Edexcel Topic 12: Warfare and British Society (c1250–present)",
    "20": "Edexcel Topic 20: Period Study - Superpower Relations and Cold War",
    "21": "Edexcel Topic 21: Period Study - The American West (c1835–c1895)",
    "30": "Edexcel Topic 30: Modern Depth Study - Weimar and Nazi Germany (1918–39)",
    "31": "Edexcel Topic 31: Modern Depth Study - Russia and the Soviet Union (1917–41)",
    "33": "Edexcel Topic 33: Modern Depth Study - USA (1954–75)",
}

def detect_subtopic(text: str, filename: str) -> str:
    combined_str = f"{filename} {text[:400]}".lower()
    
    if "medicine" in combined_str:
        return TOPIC_MAP["10"]
    elif "crime" in combined_str or "punishment" in combined_str:
        return TOPIC_MAP["11"]
    elif "cold war" in combined_str or "superpower" in combined_str:
        return TOPIC_MAP["20"]
    elif "american west" in combined_str:
        return TOPIC_MAP["21"]
    elif "weimar" in combined_str or "nazi" in combined_str:
        return TOPIC_MAP["30"]
    elif "russia" in combined_str or "soviet" in combined_str:
        return TOPIC_MAP["31"]
    
    return "Edexcel Topic 10: Medicine in Britain (c1250–present) & Western Front"

def build_vector_db():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created directory {DATA_PATH}.")
        return

    print("Loading PDF History syllabus files...")
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
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    print(f"✅ Ingestion complete! Persisted {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    build_vector_db()