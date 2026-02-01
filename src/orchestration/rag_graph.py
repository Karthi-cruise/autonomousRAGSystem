"""LangGraph orchestration: Query -> Retrieve -> Trust & Decay -> Generate -> Verify -> Update."""

from __future__ import annotations

import os
from typing import Annotated, Any, Literal, Optional, TypedDict

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.agents.retriever_agent import RetrieverAgent
from src.agents.verifier_agent import VerifierAgent
from src.agents.updater_agent import UpdaterAgent
from src.utils.schema import (
    RetrievalResult,
    VerificationResult,
    Verdict,
    RAGResponse,
)


RAG_SYSTEM_PROMPT = """You are a helpful assistant. Answer the question using ONLY the provided context.
If the context does not contain enough information, say so clearly.
Always cite your sources using [Source: <source_name>] format.
Be concise and accurate."""


class RAGState(TypedDict):
    query: str
    retrieval: Optional[RetrievalResult]
    context: str
    answer: str
    citations: list[str]
    verification: Optional[VerificationResult]
    verdict: str
    trust_explanations: list[str]
    decay_explanations: list[str]
    iteration: int
    final_response: Optional[RAGResponse]
    error: Optional[str]


def create_rag_graph(
    retriever: RetrieverAgent,
    verifier: VerifierAgent,
    updater: Optional[UpdaterAgent],
    generator_model: str = "gpt-4o-mini",
    max_retrieve_attempts: int = 2,
) -> StateGraph:
    """Build the LangGraph RAG pipeline."""

    api_key = os.environ.get("OPENAI_API_KEY")
    llm = ChatOpenAI(model=generator_model, temperature=0.1, api_key=api_key) if api_key else None

    def retrieve_node(state: RAGState) -> dict:
        """Retrieve documents with trust and decay."""
        result = retriever.retrieve(state["query"])
        context = "\n\n---\n\n".join(
            d.to_context_str() for d in result.documents
        )
        trust_explanations = []
        decay_explanations = []
        for d in result.documents:
            trust_explanations.append(
                retriever.hybrid_search.trust_scorer.explain_trust(d.metadata)
            )
            decay_explanations.append(
                retriever.hybrid_search.decay_model.explain_decay(d.metadata)
            )

        return {
            "retrieval": result,
            "context": context,
            "trust_explanations": trust_explanations,
            "decay_explanations": decay_explanations,
        }

    def generate_node(state: RAGState) -> dict:
        """Generate answer from context."""
        context = state["context"]
        query = state["query"]

        if not context:
            return {
                "answer": "I could not find relevant documents to answer this question.",
                "citations": [],
            }

        if not llm:
            first_chunk = context.split("\n\n---\n\n")[0][:500] if context else ""
            answer = f"Based on retrieved context: {first_chunk}...[Set OPENAI_API_KEY for full generation]"
        else:
            try:
                messages = [
                    SystemMessage(content=RAG_SYSTEM_PROMPT),
                    HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
                ]
                response = llm.invoke(messages)
                answer = response.content
            except Exception:
                first_chunk = context.split("\n\n---\n\n")[0][:500] if context else ""
                answer = f"Based on retrieved context: {first_chunk}...[Set OPENAI_API_KEY for full generation]"

        citations = []
        if state["retrieval"]:
            for d in state["retrieval"].documents:
                if d.metadata.source not in citations:
                    citations.append(d.metadata.source)

        return {"answer": answer, "citations": citations}

    def verify_node(state: RAGState) -> dict:
        """Verify answer for hallucinations."""
        result = verifier.verify(
            query=state["query"],
            answer=state["answer"],
            context=state["context"],
        )
        return {
            "verification": result,
            "verdict": result.verdict.value,
        }

    def increment_iteration_node(state: RAGState) -> dict:
        """Increment iteration count for re-retrieve loop."""
        return {"iteration": state.get("iteration", 0) + 1}

    def route_after_verify(state: RAGState) -> Literal["re_retrieve", "accept", "update"]:
        """Route based on verifier verdict."""
        v = state.get("verification")
        if not v:
            return "accept"
        if v.verdict == Verdict.RE_RETRIEVE and state.get("iteration", 0) < max_retrieve_attempts - 1:
            return "re_retrieve"
        if v.verdict == Verdict.FLAG_KB_ISSUE and updater:
            return "update"
        return "accept"

    def format_response_node(state: RAGState) -> dict:
        """Format final RAG response."""
        v = state.get("verification")
        verdict = Verdict.ACCEPT
        verification_explanation = ""
        if v:
            verdict = v.verdict
            verification_explanation = v.explanation

        response = RAGResponse(
            answer=state["answer"],
            citations=state.get("citations", []),
            trust_explanations=state.get("trust_explanations", []),
            decay_explanations=state.get("decay_explanations", []),
            verification_explanation=verification_explanation,
            verdict=verdict,
        )
        return {"final_response": response}

    def update_kb_node(state: RAGState) -> dict:
        """Placeholder for KB update - logged for explainability."""
        # In production: updater would ingest new docs or flag for review
        return {}

    # Build graph
    workflow = StateGraph(RAGState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("verify", verify_node)
    workflow.add_node("increment_iteration", increment_iteration_node)
    workflow.add_node("format_response", format_response_node)
    workflow.add_node("update_kb", update_kb_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "verify")
    workflow.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "re_retrieve": "increment_iteration",
            "accept": "format_response",
            "update": "update_kb",
        },
    )
    workflow.add_edge("increment_iteration", "retrieve")
    workflow.add_edge("update_kb", "format_response")

    return workflow


def run_rag(
    query: str,
    retriever: RetrieverAgent,
    verifier: VerifierAgent,
    updater: Optional[UpdaterAgent] = None,
) -> RAGResponse:
    """Run the full RAG pipeline and return final response."""
    graph_builder = create_rag_graph(retriever, verifier, updater)
    graph = graph_builder.compile()

    initial_state: RAGState = {
        "query": query,
        "retrieval": None,
        "context": "",
        "answer": "",
        "citations": [],
        "verification": None,
        "verdict": "accept",
        "trust_explanations": [],
        "decay_explanations": [],
        "iteration": 0,
        "final_response": None,
        "error": None,
    }

    final = graph.invoke(initial_state)

    resp = final.get("final_response")
    if resp:
        return resp

    v = final.get("verification")
    return RAGResponse(
        answer=final.get("answer", "No response generated."),
        citations=final.get("citations", []),
        trust_explanations=final.get("trust_explanations", []),
        decay_explanations=final.get("decay_explanations", []),
        verification_explanation=v.explanation if v else "",
        verdict=v.verdict if v else Verdict.ACCEPT,
    )
