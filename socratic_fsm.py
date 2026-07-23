import os
import re
from typing import TypedDict, Sequence, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    main_unit: Optional[str]
    sub_topic: Optional[str]
    frustration_score: float
    frustration_streak: int
    didactic_triggered: bool
    pii_blocked: bool
    turn_count: int
    last_node: str
    intent: str
    session_active: bool
    total_frustration_events: int 

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

def fetch_context(query: str, sub_topic: Optional[str]) -> str:
    """Helper to perform similarity search with optional sub_topic metadata filtering."""
    try:
        db = get_vector_db()
        if sub_topic:
            docs = db.similarity_search(query, k=2, filter={"sub_topic": sub_topic})
        else:
            docs = db.similarity_search(query, k=2)
        return "\n---\n".join([d.page_content for d in docs])
    except Exception:
        # Fallback if filter returns empty or database lacks metadata tags
        try:
            db = get_vector_db()
            docs = db.similarity_search(query, k=2)
            return "\n---\n".join([d.page_content for d in docs])
        except Exception:
            return "No specific syllabus reference retrieved."

def input_guard_node(state: AgentState) -> dict:
    last_message = state["messages"][-1].content
    if scan_for_pii(str(last_message)):
        return {
            "messages": [AIMessage(content="[SAFETY BLOCK]: Please refrain from sharing private identifiers.")],
            "pii_blocked": True,
            "intent": "PII_Violation",
            "session_active": False,
            "last_node": "TerminateNode"
        }
    return {"pii_blocked": False, "intent": "None", "session_active": True, "last_node": "input_guard"}

def socratic_tutor_node(state: AgentState) -> dict:
    last_user_message = state["messages"][-1].content
    score = calculate_frustration(str(last_user_message))
    new_streak = state.get("frustration_streak", 0) + 1 if score > 0.2 else 0
    total_frust = state.get("total_frustration_events", 0) + (1 if score > 0.2 else 0)

    target_topic = state.get("sub_topic", "General Computer Science")
    context = fetch_context(str(last_user_message), state.get("sub_topic"))

    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)
    
    system_prompt = (
        f"You are an empathetic, scaffolding Socratic Computer Science tutor focusing on: {target_topic}.\n"
        "Never supply outright answers. Formulate your response as a single, encouraging guiding question.\n"
        f"Syllabus Context:\n{context}"
    )
    
    full_history = [{"role": "system", "content": system_prompt}]
    for m in state["messages"]:
        m_type = getattr(m, "type", "")
        role = "user" if (isinstance(m, HumanMessage) or m_type == "human") else "assistant"
        msg_content = str(m.content).strip() or "[Empty message sent by user]"
        full_history.append({"role": role, "content": msg_content})
    
    response = llm.invoke(full_history)

    clean_text = response.content[0]['text'] if isinstance(response.content, list) and len(response.content) > 0 and isinstance(response.content[0], dict) else str(response.content)
    
    return {
        "messages": [AIMessage(content=clean_text)],
        "frustration_score": score,
        "frustration_streak": new_streak,
        "total_frustration_events": total_frust,
        "last_node": "socratic_tutor_node",
        "turn_count": state.get("turn_count", 0) + 1,
        "intent": "None",
        "session_active": True
    }

def didactic_fallback_node(state: AgentState) -> dict:
    last_user_message = state["messages"][-1].content
    target_topic = state.get("sub_topic", "General Computer Science")
    context = fetch_context(str(last_user_message), state.get("sub_topic"))

    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.0)
    
    system_prompt = (
        f"You have exited Socratic mode for topic: {target_topic}.\n"
        "Deliver a direct, highly structured, concise technical answer targeting ONLY "
        "the specific concept of the student's prompt. Do not dump unrelated syllabus material.\n"
        f"Syllabus Context:\n{context}"
    )
    
    full_history = [{"role": "system", "content": system_prompt}]
    for m in state["messages"]:
        m_type = getattr(m, "type", "")
        role = "user" if (isinstance(m, HumanMessage) or m_type == "human") else "assistant"
        msg_content = str(m.content).strip() or "[Empty message sent by user]"
        full_history.append({"role": role, "content": msg_content})
    
    response = llm.invoke(full_history)
    clean_text = response.content[0]['text'] if isinstance(response.content, list) and len(response.content) > 0 and isinstance(response.content[0], dict) else str(response.content)

    return {
        "messages": [AIMessage(content=clean_text)],
        "frustration_score": 0.0,
        "frustration_streak": 0,
        "total_frustration_events": state.get("total_frustration_events", 0),
        "last_node": "AutomatedFadeOut",
        "turn_count": state.get("turn_count", 0) + 1,
        "intent": "None",
        "session_active": True
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