"""LangGraph orchestration: Query -> Retrieve -> Trust & Decay -> Generate -> Verify -> Update."""

from __future__ import annotations

import os
import re
from typing import Any, Literal, Optional, TypedDict

try:
    from langgraph.graph import StateGraph
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
except ImportError:
    StateGraph = None
    ChatOpenAI = None
    SystemMessage = None
    HumanMessage = None

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


EXTRACTIVE_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "can",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "into",
    "is",
    "must",
    "of",
    "our",
    "should",
    "the",
    "their",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "your",
}


def _terms(text: str) -> set[str]:
    """Tokenize text into comparable query/content terms."""
    return {
        token
        for token in re.findall(r"\b[a-z0-9][a-z0-9_-]*\b", text.lower())
        if len(token) > 2 and token not in EXTRACTIVE_STOPWORDS
    }


def _passages(content: str) -> list[str]:
    """Split retrieved content into compact passages for offline generation."""
    blocks = re.split(r"\n\s*\n+", content)
    cleaned = []
    for block in blocks:
        block = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", block.strip())
        block = re.sub(r"(?m)^\s*[-*]\s+", "", block)
        block = re.sub(r"\s+", " ", block).strip()
        if block:
            cleaned.append(block)
    return cleaned


def _extractive_answer(query: str, retrieval: Optional[RetrievalResult]) -> str:
    """Generate a grounded answer from retrieved passages without an LLM."""
    if not retrieval or not retrieval.documents:
        return "I could not find relevant documents to answer this question."

    query_terms = _terms(query)
    candidates = []
    for doc_rank, doc in enumerate(retrieval.documents):
        for passage_rank, passage in enumerate(_passages(doc.content)):
            passage_terms = _terms(passage)
            overlap = query_terms & passage_terms
            phrase_bonus = sum(
                1
                for term in query_terms
                if term in passage.lower()
            )
            score = (len(overlap) * 2) + phrase_bonus + doc.effective_score
            if overlap:
                candidates.append((
                    score,
                    doc.effective_score,
                    -doc_rank,
                    -passage_rank,
                    passage,
                    doc.metadata.source,
                ))

    if not candidates:
        first = retrieval.documents[0]
        passage = _passages(first.content)[0][:600]
        return (
            "I found related context, but it does not directly answer the question. "
            f"Most relevant excerpt: {passage} [Source: {first.metadata.source}]"
        )

    candidates.sort(reverse=True)
    selected = []
    seen_passages = set()
    seen_sources = set()
    for _, _, _, _, passage, source in candidates:
        key = passage.lower()
        if key in seen_passages or source in seen_sources:
            continue
        selected.append(f"{passage} [Source: {source}]")
        seen_passages.add(key)
        seen_sources.add(source)
        if len(selected) >= 2:
            break

    return " ".join(selected)


class RAGState(TypedDict):
    query: str
    retrieval: Optional[RetrievalResult]
    context: str
    answer: str
    citations: list[str]
    verification: Optional[VerificationResult]
    verdict: str
    actions: list[dict]
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
    if StateGraph is None:
        raise RuntimeError("LangGraph is not installed. Use run_rag() direct pipeline instead.")

    api_key = os.environ.get("OPENAI_API_KEY")
    llm = ChatOpenAI(model=generator_model, temperature=0.1, api_key=api_key) if api_key and ChatOpenAI else None

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
            answer = _extractive_answer(query, state["retrieval"])
        else:
            try:
                messages = [
                    SystemMessage(content=RAG_SYSTEM_PROMPT),
                    HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
                ]
                response = llm.invoke(messages)
                answer = response.content
            except Exception:
                answer = _extractive_answer(query, state["retrieval"])

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
            actions=state.get("actions", []),
        )
        return {"final_response": response}

    def update_kb_node(state: RAGState) -> dict:
        """Handle verifier-raised KB issues with rollback-safe maintenance."""
        if not updater:
            return {"actions": []}

        verification = state.get("verification")
        reason = verification.explanation if verification else "Verifier flagged a KB issue."
        action = updater.handle_verification_failure(
            query=state["query"],
            answer=state["answer"],
            reason=reason,
        )
        return {"actions": [action]}

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
    generator_model: str = "gpt-4o-mini",
    max_retrieve_attempts: int = 2,
) -> RAGResponse:
    """Run the full RAG pipeline and return final response."""
    if os.environ.get("LIGHTWEIGHT_DEPLOY") == "1" or StateGraph is None:
        return run_rag_direct(
            query=query,
            retriever=retriever,
            verifier=verifier,
            updater=updater,
        )

    graph_builder = create_rag_graph(
        retriever,
        verifier,
        updater,
        generator_model=generator_model,
        max_retrieve_attempts=max_retrieve_attempts,
    )
    graph = graph_builder.compile()

    initial_state: RAGState = {
        "query": query,
        "retrieval": None,
        "context": "",
        "answer": "",
        "citations": [],
        "verification": None,
        "verdict": "accept",
        "actions": [],
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
        actions=final.get("actions", []),
    )


def _build_context_and_explanations(
    retriever: RetrieverAgent,
    retrieval: RetrievalResult,
) -> tuple[str, list[str], list[str]]:
    context = "\n\n---\n\n".join(d.to_context_str() for d in retrieval.documents)
    trust_explanations = [
        retriever.hybrid_search.trust_scorer.explain_trust(d.metadata)
        for d in retrieval.documents
    ]
    decay_explanations = [
        retriever.hybrid_search.decay_model.explain_decay(d.metadata)
        for d in retrieval.documents
    ]
    return context, trust_explanations, decay_explanations


def _generate_answer_direct(query: str, context: str, retrieval: RetrievalResult) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return _extractive_answer(query, retrieval)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.1,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
        )
        return response.choices[0].message.content or _extractive_answer(query, retrieval)
    except Exception:
        return _extractive_answer(query, retrieval)


def run_rag_direct(
    query: str,
    retriever: RetrieverAgent,
    verifier: VerifierAgent,
    updater: Optional[UpdaterAgent] = None,
) -> RAGResponse:
    """Low-memory RAG pipeline for free-tier deployments."""
    retrieval = retriever.retrieve(query)
    context, trust_explanations, decay_explanations = _build_context_and_explanations(
        retriever,
        retrieval,
    )
    answer = (
        _generate_answer_direct(query, context, retrieval)
        if context
        else "I could not find relevant documents to answer this question."
    )
    citations = []
    for doc in retrieval.documents:
        if doc.metadata.source not in citations:
            citations.append(doc.metadata.source)

    verification = verifier.verify(query=query, answer=answer, context=context)
    actions: list[dict[str, Any]] = []
    if verification.verdict == Verdict.FLAG_KB_ISSUE and updater:
        actions.append(updater.handle_verification_failure(
            query=query,
            answer=answer,
            reason=verification.explanation,
        ))

    return RAGResponse(
        answer=answer,
        citations=citations,
        trust_explanations=trust_explanations,
        decay_explanations=decay_explanations,
        verification_explanation=verification.explanation,
        verdict=verification.verdict,
        actions=actions,
    )
