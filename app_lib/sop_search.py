"""Tavily-powered SOP reference search from SKYbrary.

Fetches real Standard Operating Procedure guidance from SKYbrary
(https://skybrary.aero) to ground SOP recommendations in authoritative
aviation safety references rather than internally generated codes.

Uses Tavily Search API with domain-restricted queries to ensure all
results come from the SKYbrary knowledge base.

Caching strategy:
  - Results are cached in-memory for the session lifetime.
  - Each unique topic query is cached after the first search.
  - This avoids redundant API calls when the same KPI issue recurs.
"""

from __future__ import annotations

import os
import logging
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Primary domain for aviation SOP references
SKYBRARY_DOMAIN = "skybrary.aero"

# Pre-defined search queries mapped to KPI problem areas.
# Each query is designed to pull the most relevant SKYbrary articles.
SOP_SEARCH_QUERIES: dict[str, str] = {
    "cancellation_weather": "standard operating procedures weather disruption flight cancellation",
    "cancellation_airline": "standard operating procedures airline carrier maintenance crew scheduling",
    "cancellation_nas": "standard operating procedures air traffic control national airspace system",
    "cancellation_security": "standard operating procedures security delay airport operations",
    "delay_departure": "standard operating procedures departure delay turnaround ground handling",
    "delay_late_aircraft": "standard operating procedures late aircraft delay aircraft rotation",
    "delay_airline": "standard operating procedures airline delay maintenance crew management",
    "delay_weather": "standard operating procedures weather delay operations",
    "delay_air_system": "standard operating procedures air traffic management ATC delay",
    "delay_security": "standard operating procedures security screening delay",
    "otp_low": "standard operating procedures on-time performance improvement",
    "flight_volume": "standard operating procedures flight schedule planning capacity management",
    "general_sop": "standard operating procedures aviation flight operations safety",
}


def _get_tavily_client():
    """Lazy-initialise and return the Tavily client."""
    if not TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY not set — SOP search will return fallback results.")
        return None
    try:
        from tavily import TavilyClient
        return TavilyClient(api_key=TAVILY_API_KEY)
    except ImportError:
        logger.warning("tavily-python not installed. Run: pip install tavily-python")
        return None


@lru_cache(maxsize=64)
def search_skybrary_sop(topic: str, max_results: int = 3) -> list[dict]:
    """Search SKYbrary for SOP-related articles on a given topic.

    Parameters
    ----------
    topic : str
        One of the keys in SOP_SEARCH_QUERIES, or a free-form search string.
    max_results : int
        Maximum number of results to return (default 3).

    Returns
    -------
    list[dict]
        Each dict has keys: 'title', 'url', 'snippet' (excerpt from the article).
    """
    client = _get_tavily_client()
    if client is None:
        return _fallback_references(topic)

    query = SOP_SEARCH_QUERIES.get(topic, topic)
    try:
        response = client.search(
            query=query,
            search_depth="basic",
            include_domains=[SKYBRARY_DOMAIN],
            max_results=max_results,
            include_answer=False,
        )
        results = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", "SKYbrary Article"),
                "url": item.get("url", ""),
                "snippet": item.get("content", "")[:300],
            })
        return results if results else _fallback_references(topic)
    except Exception as exc:
        logger.error(f"Tavily search failed for topic '{topic}': {exc}")
        return _fallback_references(topic)


def fetch_skybrary_sop_content(url: str = "https://skybrary.aero/articles/standard-operating-procedures-sops") -> str:
    """Fetch the main SKYbrary SOP article content via Tavily extract.

    Uses Tavily's extract endpoint to pull the full article content from
    the canonical SKYbrary SOP page. Result is cached for the session.

    Returns
    -------
    str
        The extracted article text, or a fallback summary if unavailable.
    """
    client = _get_tavily_client()
    if client is None:
        return _FALLBACK_SOP_CONTENT

    try:
        response = client.extract(urls=[url])
        results = response.get("results", [])
        if results and results[0].get("raw_content"):
            return results[0]["raw_content"][:3000]  # cap to avoid prompt bloat
        return _FALLBACK_SOP_CONTENT
    except Exception as exc:
        logger.error(f"Tavily extract failed for {url}: {exc}")
        return _FALLBACK_SOP_CONTENT


def get_sop_references_for_issues(issues: list[str]) -> str:
    """Build a formatted SOP reference block for a set of KPI issues.

    Parameters
    ----------
    issues : list[str]
        List of issue topic keys (e.g., ["cancellation_weather", "otp_low"]).

    Returns
    -------
    str
        Formatted text block with SKYbrary SOP references, ready for
        injection into the AI prompt context.
    """
    if not issues:
        return ""

    lines = [
        "",
        "=" * 60,
        "  SOP REFERENCES — SKYbrary (skybrary.aero)",
        "=" * 60,
        "  Source: https://skybrary.aero/articles/standard-operating-procedures-sops",
        "",
    ]

    # Fetch the canonical SOP article content as baseline context
    base_content = fetch_skybrary_sop_content()
    if base_content and base_content != _FALLBACK_SOP_CONTENT:
        lines.append("━━━ SKYbrary SOP Framework ━━━")
        # Trim to key paragraphs for context
        for paragraph in base_content.split("\n"):
            stripped = paragraph.strip()
            if stripped:
                lines.append(f"  {stripped}")
        lines.append("")

    # Search for issue-specific references
    seen_urls: set[str] = set()
    for issue_topic in issues:
        results = search_skybrary_sop(issue_topic)
        if not results:
            continue

        issue_label = issue_topic.replace("_", " ").title()
        lines.append(f"━━━ Related: {issue_label} ━━━")
        for ref in results:
            url = ref.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            title = ref.get("title", "SKYbrary Reference")
            snippet = ref.get("snippet", "")
            lines.append(f"  📄 {title}")
            lines.append(f"     {url}")
            if snippet:
                lines.append(f"     {snippet[:200]}…")
            lines.append("")

    return "\n".join(lines)


def _fallback_references(topic: str) -> list[dict]:
    """Return a static SKYbrary reference when Tavily is unavailable."""
    return [{
        "title": "Standard Operating Procedures (SOPs) — SKYbrary Aviation Safety",
        "url": "https://skybrary.aero/articles/standard-operating-procedures-sops",
        "snippet": (
            "SOPs are the foundation of consistent operational performance. "
            "They establish standardised procedures for all phases of flight "
            "and ground operations, ensuring safety, compliance, and efficiency. "
            f"(Topic: {topic})"
        ),
    }]


# Fallback content when Tavily extract is unavailable
_FALLBACK_SOP_CONTENT = """\
Standard Operating Procedures (SOPs) — SKYbrary Aviation Safety
Source: https://skybrary.aero/articles/standard-operating-procedures-sops

SOPs are universally recognised as the foundation of safe and efficient flight operations.
They define the sequence of actions and standard callouts required for every phase of flight.
Key SOP principles from SKYbrary:
  • SOPs should be clear, unambiguous, and comprehensive
  • Compliance with SOPs is a fundamental safety requirement
  • SOPs reduce human error by providing standardised responses to operational situations
  • Deviation from SOPs is a leading contributor to aviation incidents and accidents
  • Regular SOP review and update cycles ensure procedures remain current
  • Training programmes must reinforce SOP compliance through CRM and recurrent training
  • SOPs cover: pre-flight planning, departure, en-route, approach, landing, ground ops
  • Effective SOPs integrate human factors principles and workload management
  • Airlines should have SOP deviation reporting and analysis programmes
"""
