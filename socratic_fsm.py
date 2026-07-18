# socratic_fsm.py
import os
import re
from typing import TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    frustration_score: float
    frustration_streak: int
    didactic_triggered: bool
    pii_blocked: bool

CHROMA_PATH = "./chroma_db"

def get_vector_db():
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(f"Vector database not found at {CHROMA_PATH}.")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

def scan_for_pii(text: str) -> bool:
    patterns = [
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", 
        r"(?:\+44|0)\s?\d{4}\s?\d{6}",                        
        r"[A-Z]{1,2}[0-9R][0-9A-Z]?\s?[0-9][A-Z]{2}"        
    ]
    return any(re.search(pat, text, re.IGNORECASE) for pat in patterns)

def calculate_frustration(text: str) -> float:
    indicators = ["stuck", "don't know", "hard", "difficult", "confused", "give up", "tell me"]
    text_lower = text.lower()
    matches = sum(1 for word in indicators if word in text_lower)
    return min(1.0, matches * 0.25)

def input_guard_node(state: AgentState) -> dict:
    """Evaluates inbound message compliance, blocking PII before reaching any LLM."""
    last_message = state["messages"][-1].content
    if scan_for_pii(str(last_message)):
        return {
            "messages": [AIMessage(content="[SAFETY BLOCK]: Please refrain from sharing private identifiers.")],
            "pii_blocked": True
        }
    return {"pii_blocked": False}

def socratic_tutor_node(state: AgentState) -> dict:
    last_user_message = state["messages"][-1].content
    score = calculate_frustration(str(last_user_message))
    new_streak = state.get("frustration_streak", 0) + 1 if score > 0.2 else 0

    try:
        db = get_vector_db()
        docs = db.similarity_search(str(last_user_message), k=2)
        context = "\n---\n".join([d.page_content for d in docs])
    except Exception:
        context = "No syllabus reference loaded. Rely on standard Computer Science curriculum guidance."

    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)

    system_prompt = (
        "You are an empathetic, scaffolding Socratic Computer Science tutor. "
        "Never supply the outright programming, hardware, or algorithm answers. "
        "Formulate your response as a single, encouraging guiding question.\n"
        f"Syllabus Context:\n{context}"
    )
    
    full_history = [{"role": "system", "content": system_prompt}]
    for m in state["messages"]:
        m_type = getattr(m, "type", "")
        role = "user" if (isinstance(m, HumanMessage) or m_type == "human") else "assistant"
        full_history.append({"role": role, "content": str(m.content)})
    
    response = llm.invoke(full_history)
    
    if isinstance(response.content, list) and len(response.content) > 0:
        if isinstance(response.content[0], dict) and 'text' in response.content[0]:
            clean_text = response.content[0]['text']
        else:
            clean_text = str(response.content[0])
    else:
        clean_text = str(response.content)
    
    return {
        "messages": [AIMessage(content=clean_text)],
        "frustration_score": score,
        "frustration_streak": new_streak
    }

def didactic_fallback_node(state: AgentState) -> dict:
    last_user_message = state["messages"][-1].content
    try:
        db = get_vector_db()
        docs = db.similarity_search(str(last_user_message), k=2)
        context = "\n---\n".join([d.page_content for d in docs])
    except Exception:
        context = "Standard CS syllabus rules."

    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.0)
   
    system_prompt = (
        "You have exited Socratic mode because the conversation turn limit has been met.\n"
        "Deliver a direct, highly structured, concise technical answer that targets ONLY "
        "the specific sub-topic scope of the student's immediately preceding prompt or error.\n"
        "Do not dump the entire syllabus context or unrelated network architectures.\n"
        f"Syllabus Context:\n{context}"
    )
    
    full_history = [{"role": "system", "content": system_prompt}]
    for m in state["messages"]:
        m_type = getattr(m, "type", "")
        role = "user" if (isinstance(m, HumanMessage) or m_type == "human") else "assistant"
        full_history.append({"role": role, "content": str(m.content)})
    
    response = llm.invoke(full_history)
    
    if isinstance(response.content, list) and len(response.content) > 0:
        if isinstance(response.content[0], dict) and 'text' in response.content[0]:
            clean_text = response.content[0]['text']
        else:
            clean_text = str(response.content[0])
    else:
        clean_text = str(response.content)
    
    return {
        "messages": [AIMessage(content=clean_text)],
        "frustration_streak": 0,  
        "didactic_triggered": True
    }

def gate_route(state: AgentState) -> str:
    if state.get("pii_blocked", False):
        return END
    if state.get("didactic_triggered", False):
        return "didactic_fallback"
    return "socratic_tutor"

def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("input_guard", input_guard_node)
    workflow.add_node("socratic_tutor", socratic_tutor_node)
    workflow.add_node("didactic_fallback", didactic_fallback_node)
    
    workflow.set_entry_point("input_guard")
    workflow.add_conditional_edges("input_guard", gate_route, {
        END: END,
        "socratic_tutor": "socratic_tutor",
        "didactic_fallback": "didactic_fallback"
    })
    
    workflow.add_edge("socratic_tutor", END)
    workflow.add_edge("didactic_fallback", END)
    
    return workflow.compile()