"""
LangChain Tools wrappers for Flight-Report

Provides lightweight, non-invasive wrappers around existing functions so they
can be used as callable tools by LangChain agents. This module does NOT change
any existing behavior — it simply exposes the existing capabilities in a
tool-friendly shape and attempts to construct LangChain `Tool` objects when
LangChain is available.

Defined tools (callable):
- groq_extractor(image_path) -> dict
- azure_qa(extracted_result, question, max_tokens=1024, temperature=0.0) -> dict

It also exposes `lc_tools` (list) containing LangChain `Tool` objects when
LangChain is importable; otherwise `lc_tools` is empty and `tools` dict still
contains the callables for direct use.
"""

from typing import Any, Dict
from functools import partial

try:
    # Local imports from existing codebase
    from groq_vision import extract_dashboard_data_with_groq, format_extracted_data_for_llm, send_extracted_data_to_azure
except Exception:
    # Keep functions defined but point to None if imports fail
    extract_dashboard_data_with_groq = None
    format_extracted_data_for_llm = None
    send_extracted_data_to_azure = None

try:
    from promptAI import llm
except Exception:
    llm = None

# Try to import azure_context helpers for prompt enrichment
try:
    from azure_context import append_kpi_context, get_azure_system_prompt
except Exception:
    append_kpi_context = None
    get_azure_system_prompt = None


def groq_extractor(image_path: str) -> Dict[str, Any]:
    """Extract dashboard data using existing Groq extractor.

    Returns the same dict produced by `extract_dashboard_data_with_groq`.
    """
    if extract_dashboard_data_with_groq is None:
        return {"success": False, "error": "Groq extractor not available"}
    return extract_dashboard_data_with_groq(image_path)


def azure_qa(extracted_result: Dict[str, Any], question: str, max_tokens: int = 1024, temperature: float = 0.0) -> Dict[str, Any]:
    """Ask a question of Azure OpenAI using LangChain `llm` when available.

    Behavior:
    - Build context with `format_extracted_data_for_llm` and `append_kpi_context` (if present)
    - If `promptAI.llm` is available and supports `.invoke()`, use it
    - Otherwise fall back to existing `send_extracted_data_to_azure` function
    """
    # Build context
    context = None
    if format_extracted_data_for_llm:
        try:
            context = format_extracted_data_for_llm(extracted_result)
        except Exception:
            context = None

    if append_kpi_context and context is not None:
        try:
            context = append_kpi_context(context, extracted_result)
        except Exception:
            pass

    user_prompt = f"{context}\n\nQuestion: {question}\n\nProvide a concise, factual answer based only on the data above." if context else question

    # Prefer LangChain llm if available
    if llm is not None:
        try:
            # Maintain compatibility with existing code that calls `.invoke(messages)`
            system_prompt = get_azure_system_prompt() if get_azure_system_prompt else None
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            # Call the LangChain-backed LLM; many wrappers in this repo expose `.invoke()`
            if hasattr(llm, 'invoke'):
                resp = llm.invoke(messages)
                content = getattr(resp, 'content', resp)
                return {"success": True, "answer": content, "raw_response": resp}

            # Fallback to callable llm (some LangChain chat models are callable)
            if callable(llm):
                resp = llm(messages)
                content = getattr(resp, 'content', resp)
                return {"success": True, "answer": content, "raw_response": resp}
        except Exception as e:
            # Continue to next fallback
            pass

    # Final fallback: call existing REST-based helper if present
    if send_extracted_data_to_azure is not None:
        try:
            return send_extracted_data_to_azure(extracted_result, question, max_tokens=max_tokens, temperature=temperature)
        except Exception as e:
            return {"success": False, "error": f"Azure QA failed: {e}"}

    return {"success": False, "error": "No Azure LLM integration available"}


# Expose a simple dict of tools (callables)
tools = {
    "GroqExtractorTool": groq_extractor,
    "AzureQATool": azure_qa,
}

# Try to construct LangChain `Tool` objects when LangChain is present
lc_tools = []
try:
    # Newer LangChain versions provide `Tool` in `langchain.agents` or `langchain.tools`.
    try:
        from langchain.agents import Tool
    except Exception:
        from langchain.tools import Tool

    lc_tools.append(Tool.from_function(func=groq_extractor, name="groq_extractor", description="Extract dashboard data from an image path using Groq Vision."))
    lc_tools.append(Tool.from_function(func=azure_qa, name="azure_qa", description="Ask a question of the extracted dashboard data using Azure OpenAI."))
except Exception:
    # If LangChain isn't installed or API differs, leave lc_tools empty — callers can use `tools` dict instead
    lc_tools = []


def tools_summary() -> str:
    """Return a short text summary of available tools (callable names and LangChain Tool status)."""
    lines = ["Defined tools:"]
    for k in tools.keys():
        lines.append(f" - {k}")
    lines.append("")
    if lc_tools:
        lines.append("LangChain Tool objects available:")
        for t in lc_tools:
            try:
                lines.append(f" - {t.name}: {t.description}")
            except Exception:
                lines.append(f" - {getattr(t, 'name', repr(t))}")
    else:
        lines.append("No LangChain Tool objects available (langchain not installed or API mismatch). Use `tools` dict instead.")
    return "\n".join(lines)


if __name__ == '__main__':
    print(tools_summary())
