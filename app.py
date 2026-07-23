import streamlit as st
import re
from langchain_core.messages import HumanMessage, AIMessage
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from socratic_fsm import build_graph

# --- 1. Enhanced Custom Styling ---
st.markdown("""
    <style>
    /* Background & Main Container */
    .stApp { 
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
    }
    div[data-testid="stSidebar"] { 
        background-color: #ffffff; 
        border-right: 1px solid #e2e8f0;
    }
    h1, h2, h3 { 
        color: #0f172a; 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 700;
    }
    
    /* Sleek Vibrant Header */
    .chat-header { 
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%); 
        color: white; 
        padding: 22px; 
        font-weight: 700; 
        text-align: center; 
        font-size: 1.3em; 
        border-radius: 16px; 
        box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.25);
        margin-bottom: 24px; 
        letter-spacing: 0.5px;
    }

    /* Target Selection Card Container */
    .selection-card {
        background: #ffffff;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }

    /* Message Bubble Aesthetics */
    .tutor-msg { 
        background-color: #ffffff; 
        color: #1e293b; 
        padding: 16px 20px; 
        border-radius: 18px 18px 18px 4px; 
        margin-bottom: 14px; 
        max-width: 82%; 
        line-height: 1.5; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        border: 1px solid #e2e8f0;
    }
    .student-msg { 
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%); 
        color: white; 
        padding: 16px 20px; 
        border-radius: 18px 18px 4px 18px; 
        margin-bottom: 14px; 
        max-width: 82%; 
        margin-left: auto; 
        line-height: 1.5;
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.2);
    }
    .summary-box { 
        background: #fefce8; 
        border-left: 5px solid #eab308; 
        padding: 18px; 
        border-radius: 12px; 
        color: #713f12; 
        font-size: 0.98em; 
        margin: 18px 0; 
        max-width: 85%; 
        line-height: 1.5; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }

    /* Button Styling */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out !important;
    }
    </style>
""", unsafe_allow_html=True)

# Compile LangGraph Workflow engine
graph = build_graph()

# --- 2. Paper 2 Specification Master Topics ---
AQA_PAPER_2_TOPICS = {
    "3.3 Fundamentals of Data Representation": [
        "AQA 3.3.1: Number Bases & Conversions (Binary, Hex)",
        "AQA 3.3.2: Units of Information & Binary Arithmetic",
        "AQA 3.3.3: Character Encoding (ASCII & Unicode)",
        "AQA 3.3.4: Representing Images & Sound",
        "AQA 3.3.5: Data Compression (Huffman & RLE)"
    ],
    "3.4 Computer Systems": [
        "AQA 3.4.1: Hardware & Software Classification",
        "AQA 3.4.2: Systems Architecture (CPU & Von Neumann)",
        "AQA 3.4.3: Memory (RAM, ROM & Cache)",
        "AQA 3.4.4: Secondary Storage Devices"
    ],
    "3.5 Fundamentals of Computer Networks": [
        "AQA 3.5.1: Network Types & Connections (PAN, LAN, WAN)",
        "AQA 3.5.2: Network Topologies & Data Routing (Star, Bus, Packets)",
        "AQA 3.5.3: Protocols & The 4-Layer TCP/IP Model"
    ],
    "3.6 Cyber Security": [
        "AQA 3.6.1: Cyber Security Threats & Attacks",
        "AQA 3.6.2: Social Engineering & Malware Methods",
        "AQA 3.6.3: Cyber Security Prevention & Detection"
    ],
    "3.7 Relational Databases": [
        "AQA 3.7.1: Relational Database Concepts & Structure",
        "AQA 3.7.2: Primary Keys, Foreign Keys & Relationships"
    ],
    "3.8 Ethical, Legal and Environmental Impacts": [
        "AQA 3.8.1: Ethical & Privacy Issues",
        "AQA 3.8.2: Environmental Impact of Technology",
        "AQA 3.8.3: Computer Legislation (DPA, CMA, Copyright)"
    ]
}

# --- 3. Initialize Session States ---
if "active_unit" not in st.session_state:
    st.session_state.active_unit = None
if "active_topic" not in st.session_state:
    st.session_state.active_topic = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0
if "graph_state" not in st.session_state:
    st.session_state.graph_state = {
        "messages": [],
        "main_unit": None,
        "sub_topic": None,
        "frustration_score": 0.0,
        "frustration_streak": 0,
        "didactic_triggered": False,
        "pii_blocked": False
    }

def reset_session():
    st.session_state.turn_count = 0
    st.session_state.active_unit = None
    st.session_state.active_topic = None
    st.session_state.messages = []
    st.session_state.graph_state = {
        "messages": [],
        "main_unit": None,
        "sub_topic": None,
        "frustration_score": 0.0,
        "frustration_streak": 0,
        "didactic_triggered": False,
        "pii_blocked": False
    }
    st.rerun()

# --- 4. Single-Screen View Router ---
if st.session_state.active_topic is None:
    st.markdown('<div class="chat-header">⚡ AQA 8525 Paper 2 Socratic Coach</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="selection-card">', unsafe_allow_html=True)
    st.subheader("🎯 Select Revision Target")
    st.write("Choose a topic module below to launch your guided Socratic practice session:")
    
    # Tier 1: Main Unit Selection
    selected_unit = st.selectbox(
        "📘 Step 1: Choose Main Unit:",
        options=list(AQA_PAPER_2_TOPICS.keys())
    )
    
    # Tier 2: Sub-Topic Selection
    selected_subtopic = st.selectbox(
        "🔍 Step 2: Choose Specific Sub-Topic:",
        options=AQA_PAPER_2_TOPICS[selected_unit]
    )
    
    st.write("")
    if st.button("🚀 Start Socratic Session", type="primary", use_container_width=True):
        st.session_state.active_unit = selected_unit
        st.session_state.active_topic = selected_subtopic
        st.session_state.graph_state["main_unit"] = selected_unit
        st.session_state.graph_state["sub_topic"] = selected_subtopic
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

else:
    topic_code = st.session_state.active_topic.split(":")[0]
    st.markdown(f'<div class="chat-header">🎓 Socratic Coach ({topic_code})</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.subheader("📌 Active Target")
        st.info(f"**Unit:** {st.session_state.active_unit}\n\n**Topic:** {st.session_state.active_topic}")
        
        st.metric(label="Turn Counter", value=f"{st.session_state.turn_count} / 5")
        st.progress(st.session_state.turn_count / 5)
        
        st.write("---")
        if st.button("🔄 New Session / Change Topic", use_container_width=True):
            reset_session()

    if len(st.session_state.messages) == 0:
        initial_greeting = f"Hi there! Let's review **{st.session_state.active_topic}** today. What is the first thing that comes to mind when you hear about this topic?"
        st.session_state.messages.append({"role": "tutor", "content": initial_greeting, "style": "tutor-msg"})
        st.session_state.graph_state["messages"].append(AIMessage(content=initial_greeting))

    for msg in st.session_state.messages:
        if msg["role"] == "tutor":
            div_class = msg.get("style", "tutor-msg")
            header = "💡 <b>Summary Note</b>" if div_class == "summary-box" else "🤖 <b>Socratic Coach</b>"
            st.markdown(f'<div class="{div_class}">{header}<br>{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="student-msg">🎒 <b>Student</b><br>{msg["content"]}</div>', unsafe_allow_html=True)

    if user_input := st.chat_input("Type your response here..."):
        st.session_state.messages.append({"role": "student", "content": user_input})
        st.session_state.graph_state["messages"].append(HumanMessage(content=user_input))
        
        st.session_state.turn_count += 1
        
        if st.session_state.turn_count >= 5:
            st.session_state.graph_state["didactic_triggered"] = True
        else:
            st.session_state.graph_state["didactic_triggered"] = False
        
        with st.spinner("Analyzing response..."):
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

        display_style = "summary-box" if updated_state.get("didactic_triggered", False) else "tutor-msg"
        
        st.session_state.messages.append({"role": "tutor", "content": ai_reply, "style": display_style})
        st.session_state.graph_state = updated_state
        st.rerun()