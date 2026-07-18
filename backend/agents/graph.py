"""Assembles the LangGraph StateGraph wiring all six navigator agents."""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from backend.agents.document_agent import run_document_agent
from backend.agents.eligibility_agent import run_eligibility_agent
from backend.agents.intake_agent import run_intake_agent
from backend.agents.retrieval_agent import run_retrieval_agent
from backend.agents.simplification_agent import run_simplification_agent
from backend.agents.state import AgentState
from backend.agents.traceability_agent import run_traceability_agent


def build_navigator_graph():
    graph = StateGraph(AgentState)

    graph.add_node("intake", run_intake_agent)
    graph.add_node("retrieval", run_retrieval_agent)
    graph.add_node("eligibility", run_eligibility_agent)
    graph.add_node("document_guidance", run_document_agent)
    graph.add_node("simplification", run_simplification_agent)
    graph.add_node("traceability", run_traceability_agent)

    graph.set_entry_point("intake")
    graph.add_edge("intake", "retrieval")
    graph.add_edge("retrieval", "eligibility")
    graph.add_edge("eligibility", "document_guidance")
    graph.add_edge("document_guidance", "simplification")
    graph.add_edge("simplification", "traceability")
    graph.add_edge("traceability", END)

    return graph.compile()


_compiled_graph = None


def get_navigator_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_navigator_graph()
    return _compiled_graph
