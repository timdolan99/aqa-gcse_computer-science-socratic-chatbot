#!/usr/bin/env python3
"""
evaluate_rag.py

Automated LLM-as-a-Judge benchmarking harness for RAG performance evaluation.
Computes Context Relevance and Context Groundedness without human interaction.
"""

from __future__ import annotations
import os
import re
import tomllib
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Inject Secrets into Environment Variables ---
secrets_path = os.path.join(".streamlit", "secrets.toml")
if os.path.exists(secrets_path):
    with open(secrets_path, "rb") as f:
        secrets = tomllib.load(f)
        if "GOOGLE_API_KEY" in secrets:
            os.environ["GOOGLE_API_KEY"] = secrets["GOOGLE_API_KEY"]

# Instantiate highly deterministic judge model
judge_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.0)

def clean_float_extraction(text: str) -> float:
    """Safely extracts a single float value from text, clearing conversational noise or markdown blocks."""
    cleaned = text.strip().replace("```json", "").replace("```markdown", "").replace("```", "")
    match = re.search(r"[-+]?\d*\.\d+|\d+", cleaned)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return 0.0
    return 0.0

def score_context_relevance(query: str, chunks: list[str]) -> float:
    """Evaluates semantic alignment between query and fetched material."""
    if not chunks:
        return 0.0
    
    context_str = "\n---\n".join(chunks)
    prompt = f"""
    You are an automated academic software evaluator. Assess the Context Relevance of this RAG retrieval phase.
    
    User Query: "{query}"
    Retrieved Context Material:
    \"\"\"
    {context_str}
    \"\"\"
    
    Evaluate if the retrieved context contains information directly pertinent to answering or scaffolding the user query.
    Respond with EXACTLY one number based on this strict scale:
    1.0 = Fully relevant; the exact concepts needed are present.
    0.5 = Partially relevant; relates to the general topic but lacks specific precision.
    0.0 = Entirely irrelevant noise or empty context.
    
    Output format: ONLY output the float number (e.g., 1.0 or 0.5). Do not write anything else.
    """
    try:
        response = judge_llm.invoke(prompt)
        return clean_float_extraction(str(response.content))
    except Exception:
        return 0.0

def score_context_groundedness(response_text: str, chunks: list[str]) -> float:
    """Evaluates if the tutor's output stays grounded inside the context."""
    if not chunks:
        return 0.0
        
    context_str = "\n---\n".join(chunks)
    prompt = f"""
    You are an automated academic software evaluator. Assess the Groundedness (Faithfulness) of the system output.
    
    Retrieved Context Base:
    \"\"\"
    {context_str}
    \"\"\"
    
    Generated System Response: "{response_text}"
    
    Determine if the concepts or guidance shared in the System Response are completely supported by and derived from the Retrieved Context Base.
    If the response introduces facts, rules, or curriculum information NOT explicitly outlined in the context, it is ungrounded.
    
    Respond with EXACTLY one number based on this strict scale:
    1.0 = Fully grounded; every implied technical fact is supported by the context.
    0.5 = Partially grounded; some facts match, but it adds outside curriculum assumptions.
    0.0 = Completely ungrounded hallucination or fabrication relative to this text block.
    
    Output format: ONLY output the float number. Do not write anything else.
    """
    try:
        response = judge_llm.invoke(prompt)
        return clean_float_extraction(str(response.content))
    except Exception:
        return 0.0

if __name__ == "__main__":
    print("=== RUNNING RAG EVALUATION PROTOCOL ===")
    
    # Test cases mirroring your benchmark scenarios
    test_scenarios = [
        {
            "id": "Valid Retrieval Path",
            "query": "What is the purpose of a MAC address?",
            "chunks": ["MAC addresses are unique hardware identifiers assigned by manufacturers for local network communications."],
            "response": "Think about your computer hardware—what unique identifier does a network switch check to find your physical machine?"
        },
        {
            "id": "Irrelevant / Noisy Retrieval Path",
            "query": "What is the purpose of a MAC address?",
            "chunks": ["Bus topology connects every device to a single shared backbone cable with terminators at both ends."],
            "response": "How do data packets find the physical network card on your specific machine?"
        }
    ]
    
    print(f"{'Scenario ID':<33} | {'Relevance Score':<15} | {'Groundedness Score':<18}")
    print("-" * 74)
    for s in test_scenarios:
        rel = score_context_relevance(s["query"], s["chunks"])
        grd = score_context_groundedness(s["response"], s["chunks"])
        print(f"{s['id']:<33} | {rel:<15.1f} | {grd:<18.1f}")
