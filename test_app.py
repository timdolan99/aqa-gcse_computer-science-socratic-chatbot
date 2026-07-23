import streamlit as st
import re
from langchain_core.messages import HumanMessage, AIMessage
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from socratic_fsm import build_graph

# --- 1. Global Custom Styling ---
st.markdown("""
    <style>
    .stApp { background-color: #f0f2f5; }
    div[data-testid="stSidebar"] { background-color: #ffffff; }
    h1, h2, h3 { color: #1e1b4b; font-family: 'Segoe UI', sans-serif; }
    .chat-header { background: #4f46e5; color: white; padding: 18px; font-weight: bold; text-align: center; font-size: 1.15em; border-radius: 12px; margin-bottom: 20px; }
    .tutor-msg { background-color: #e0e7ff; color: #1e1b4b; padding: 12px 16px; border-radius: 16px 16px 16px 4px; margin-bottom: 12px; max-width: 80%; line-height: 1.45; }
    .student-msg { background-color: #4f46e5; color: white; padding: 12px 16px; border-radius: 16px 16px 4px 16px; margin-bottom: 12px; max-width: 80%; margin-left: auto; }
    .summary-box { background: #fef08a; border-left: 4px solid #eab308; padding: 15px; border-radius: 8px; color: #713f12; font-size: 0.95em; margin-top: 15px; margin-bottom: 15px; max-width: 80%; line-height: 1.45; }
    </style>
""", unsafe_allow_html=True)

# Compile LangGraph Workflow engine
graph = build_graph()

# --- 2. Initialize Session States ---
if "active_topic" not in st.session_state:
    st.session_state.active_topic = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0
if "graph_state" not in st.session_state:
    st.session_state.graph_state = {
        "messages": [],
        "frustration_score": 0.0,
        "frustration_streak": 0,
        "didactic_triggered": False,
        "pii_blocked": False
    }

def reset_session():
    st.session_state.turn_count = 0
    st.session_state.active_topic = None
    st.session_state.messages = []
    st.session_state.graph_state = {
        "messages": [],
        "frustration_score": 0.0,
        "frustration_streak": 0,
        "didactic_triggered": False,
        "pii_blocked": False
    }
    st.rerun()

# --- 3. View Router ---
if st.session_state.active_topic is None:
    st.markdown('<div class="chat-header">🎓 AQA 8525 Socratic Tutor Sandbox</div>', unsafe_allow_html=True)
    st.subheader("Select an AQA Specification Target")
    st.write("Choose a curriculum module block below to launch your interactive revision session:")
    
    detected_topics = set()
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
        db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        all_docs = db.get()
        
        for doc_meta, doc_text in zip(all_docs.get('metadatas', []), all_docs.get('documents', [])):
            match = re.search(r'(AQA\s*3\.5\.\d+[^:\n\r]*)', doc_text)
            if match:
                detected_topics.add(match.group(1).strip())
    except Exception:
        pass

    topics = sorted(list(detected_topics))

    if not topics:
        topics = [
            "AQA 3.5.1: Network Types & Connections (PAN, LAN, WAN)",
            "AQA 3.5.2: Network Topologies & Data Routing (Star, Bus, Packets)",
            "AQA 3.5.3: Protocols & The 4-Layer TCP/IP Model"
        ]

    for idx, topic in enumerate(topics, 1):
        if st.button(f"[{idx}] {topic}", use_container_width=True):
            st.session_state.active_topic = topic
            st.rerun()

else:
    topic_code = st.session_state.active_topic.split(":")[0]
    st.markdown(f'<div class="chat-header">🎓 Socratic Coach ({topic_code})</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.write(f"**Current Module:**\n{st.session_state.active_topic}")
        st.write(f"**Exchange Turn Count:** {st.session_state.turn_count} / 5")
        
        if st.button("New Session", type="primary"):
            reset_session()

    if len(st.session_state.messages) == 0:
        initial_greeting = f"Hi there! Let's review {st.session_state.active_topic} today. What is the first thing that comes to mind when you hear about this topic?"
        st.session_state.messages.append({"role": "tutor", "content": initial_greeting, "style": "tutor-msg"})
        st.session_state.graph_state["messages"].append(AIMessage(content=initial_greeting))

    for msg in st.session_state.messages:
        if msg["role"] == "tutor":
            div_class = msg.get("style", "tutor-msg")
            header = "<b>🤖 Summary Note</b>" if div_class == "summary-box" else "<b>🤖 Tutor (Gemini)</b>"
            st.markdown(f'<div class="{div_class}">{header}<br>{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="student-msg"><b>🎒 Student</b><br>{msg["content"]}</div>', unsafe_allow_html=True)

    if user_input := st.chat_input("Type your answer here..."):
        st.session_state.messages.append({"role": "student", "content": user_input})
        st.session_state.graph_state["messages"].append(HumanMessage(content=user_input))
        
        # Increment only after the baseline question response loop is committed
        st.session_state.turn_count += 1
        
        if st.session_state.turn_count >= 5:
            st.session_state.graph_state["didactic_triggered"] = True
        else:
            st.session_state.graph_state["didactic_triggered"] = False
        
        with st.spinner("Thinking..."):
            updated_state = graph.invoke(st.session_state.graph_state)
        
        last_msg = updated_state["messages"][-1]
        
        if hasattr(last_msg, "content"):
            raw_reply = last_msg.content
        elif isinstance(last_msg, dict) and "content" in last_msg:
            raw_reply = last_msg["content"]
        else:
            raw_reply = last_msg

        if isinstance(raw_reply, list) and len(raw_reply) > 0:
            if isinstance(raw_reply[0], dict) and 'text' in raw_reply[0]:
                ai_reply = raw_reply[0]['text']
            elif hasattr(raw_reply[0], 'text'):
                ai_reply = raw_reply[0].text
            else:
                ai_reply = str(raw_reply[0])
        elif isinstance(raw_reply, dict) and 'text' in raw_reply:
            ai_reply = raw_reply['text']
        else:
            ai_reply = str(raw_reply)

        # Style mapping conditionally for didactic fallback transitions
        display_style = "summary-box" if updated_state.get("didactic_triggered", False) else "tutor-msg"
        
        st.session_state.messages.append({"role": "tutor", "content": ai_reply, "style": display_style})
        st.session_state.graph_state = updated_state
        st.rerun()