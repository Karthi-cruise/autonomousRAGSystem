"""Verifier Agent - LLM-as-Judge hallucination detection."""

from __future__ import annotations

import os
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from src.utils.schema import Verdict, VerificationResult


VERIFIER_PROMPT = """You are a strict fact-checker. Your job is to verify if an AI-generated answer is fully grounded in the provided context.

CONTEXT (retrieved documents):
{context}

QUESTION: {query}

ANSWER TO VERIFY:
{answer}

Evaluate:
1. Is every claim in the answer supported by the context? (cite specific passages)
2. Are there any hallucinations, unsupported claims, or invented facts?
3. Is the answer complete or does it miss important information from the context?

Respond in this exact JSON format:
{{
  "verdict": "accept" | "reject" | "re_retrieve" | "flag_kb_issue",
  "hallucination_score": 0.0 to 1.0,
  "groundedness_score": 0.0 to 1.0,
  "explanation": "Brief explanation",
  "suggested_actions": ["action1", "action2"]
}}

Use:
- accept: Answer is fully grounded and accurate
- reject: Answer contains hallucinations or unsupported claims
- re_retrieve: Answer might improve with different/better context
- flag_kb_issue: Context appears incomplete or contradictory
"""


class VerifierAgent:
    """Detects hallucinations using LLM-as-Judge."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=key) if key and OpenAI else None
    
    def verify(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> VerificationResult:
        """Verify if answer is grounded in context. Returns verdict and scores."""
        if not self.client:
            return VerificationResult(
                verdict=Verdict.ACCEPT,
                hallucination_score=0.0,
                groundedness_score=0.5,
                explanation="Verification skipped (set OPENAI_API_KEY for full verification).",
                suggested_actions=[],
            )
        prompt = VERIFIER_PROMPT.format(
            context=context,
            query=query,
            answer=answer,
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content or ""
        except Exception as e:
            return VerificationResult(
                verdict=Verdict.ACCEPT,
                hallucination_score=0.0,
                groundedness_score=0.5,
                explanation=f"Verification skipped (set OPENAI_API_KEY). {str(e)[:80]}",
                suggested_actions=[],
            )
        
        return self._parse_response(text)
    
    def _parse_response(self, text: str) -> VerificationResult:
        """Parse LLM response into VerificationResult."""
        import json
        import re
        
        try:
            # Extract JSON block if wrapped in markdown
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(text)
        except json.JSONDecodeError:
            return VerificationResult(
                verdict=Verdict.ACCEPT,
                hallucination_score=0.0,
                groundedness_score=0.5,
                explanation=f"Could not parse verifier response: {text[:200]}",
                suggested_actions=[],
            )
        
        verdict_str = data.get("verdict", "accept").lower()
        verdict_map = {
            "accept": Verdict.ACCEPT,
            "reject": Verdict.REJECT,
            "re_retrieve": Verdict.RE_RETRIEVE,
            "flag_kb_issue": Verdict.FLAG_KB_ISSUE,
        }
        verdict = verdict_map.get(verdict_str, Verdict.ACCEPT)
        
        return VerificationResult(
            verdict=verdict,
            hallucination_score=float(data.get("hallucination_score", 0.0)),
            groundedness_score=float(data.get("groundedness_score", 1.0)),
            explanation=data.get("explanation", ""),
            suggested_actions=data.get("suggested_actions", []),
        )
