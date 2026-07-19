from __future__ import annotations
import os
import sys
import pytest
from langchain_core.messages import HumanMessage, AIMessage

# Ensure local imports work seamlessly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import socratic_fsm
from socratic_fsm import (
    build_graph, 
    AgentState, 
    input_guard_node, 
    socratic_tutor_node, 
    didactic_fallback_node,
    calculate_frustration,
    scan_for_pii
)
from evaluate_system import PROFILES, run_profile, run_all

# =====================================================================
# 1. METRICS & HEURISTICS TESTS
# =====================================================================

def test_calculate_frustration_increments_correctly():
    assert calculate_frustration("I am confused and stuck") == 0.5
    assert calculate_frustration("clear calm response") == 0.0
    assert calculate_frustration("stuck stuck stuck stuck give up") == 0.5

def test_scan_for_pii_detections():
    assert scan_for_pii("my email is test@school.ac.uk") is True
    assert scan_for_pii("call me on 07911123456") is True
    assert scan_for_pii("Normal curriculum question") is False

# =====================================================================
# 2. ISOLATED NODE BEHAVIOR TESTS
# =====================================================================

def test_input_guard_node_intercepts_explicit_pii():
    state: AgentState = {
        "messages": [HumanMessage(content="My phone number is 07911123456")],
        "frustration_score": 0.0,
        "frustration_streak": 0,
        "didactic_triggered": False,
        "pii_blocked": False,
        "turn_count": 0,
        "last_node": "None",
        "intent": "None",
        "session_active": True,
        "total_frustration_events": 0
    }
    
    delta = input_guard_node(state)
    assert delta["pii_blocked"] is True
    assert delta["intent"] == "PII_Violation"
    assert delta["session_active"] is False
    assert "[SAFETY BLOCK]" in delta["messages"][0].content

def test_input_guard_node_passes_compliant_input():
    state: AgentState = {
        "messages": [HumanMessage(content="What is a protocol?")],
        "frustration_score": 0.0,
        "frustration_streak": 0,
        "didactic_triggered": False,
        "pii_blocked": False,
        "turn_count": 0,
        "last_node": "None",
        "intent": "None",
        "session_active": True,
        "total_frustration_events": 0
    }
    
    delta = input_guard_node(state)
    assert delta["pii_blocked"] is False
    assert delta["intent"] == "None"

# =====================================================================
# 3. EVALUATION HARNESS ARCHITECTURE TESTS
# =====================================================================

def test_exactly_20_profiles_defined():
    assert len(PROFILES) == 20

def test_all_profile_ids_are_unique():
    ids = [p["id"] for p in PROFILES]
    assert len(ids) == len(set(ids))

def test_every_profile_has_required_fields():
    for p in PROFILES:
        assert "id" in p and "category" in p and "rationale" in p and "turns" in p
        assert isinstance(p["turns"], list) and len(p["turns"]) >= 1