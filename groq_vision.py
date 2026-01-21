"""
Groq Vision Module
Uses Groq's Llama-4-Scout vision model to extract structured data from dashboard screenshots.
"""

import os
import base64
import json
import requests
from dotenv import load_dotenv

load_dotenv()
from azure_context import append_kpi_context, get_azure_system_prompt

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Azure OpenAI (Azure AI) configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")


def encode_image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_dashboard_data_with_groq(image_path: str) -> dict:
    """
    Use Groq's Llama-4-Scout vision model to extract structured data from dashboard screenshot.
    
    Args:
        image_path: Path to the dashboard screenshot
        
    Returns:
        Dictionary containing extracted dashboard data
    """
    base64_image = encode_image_to_base64(image_path)
    
    extraction_prompt = """You are an expert data analyst examining a Power BI flight data dashboard screenshot.
Your task is to extract ALL visible data with MAXIMUM precision and detail.

CRITICAL EXTRACTION RULES:
1. Extract EXACT numbers as shown (e.g., if you see "0.32%" write "0.32%", if you see "0.003198" write "0.003198")
2. For slicers/filters: Extract the EXACT selected values (date ranges, airline names, etc.)
3. For charts: Extract the actual DATA VALUES shown, not just descriptions
4. Read all visible text, numbers, labels, and legends carefully
5. If a value is unclear, mark it as "unclear" but try your best first

Return a JSON object with this EXACT structure:
{
    "kpis": {
        "total_flights": "<exact number as shown>",
        "cancellation_rate": "<exact value as shown, preserve format like 0.32% or 0.003198>",
        "on_time_rate": "<exact percentage as shown>",
        "average_delay": "<exact value with unit as shown>",
        "total_airlines": "<number if visible>",
        "total_airports": "<number if visible>",
        "other_kpis": [{"name": "<kpi name>", "value": "<exact value>"}]
    },
    "slicers_and_filters": {
        "date_range": {
            "start_date": "<exact start date if visible, e.g., '1/1/2015'>",
            "end_date": "<exact end date if visible, e.g., '12/31/2015'>",
            "full_range_text": "<full text if shown as single string>"
        },
        "airline_filter": {
            "selected_airlines": ["<list of selected airline names>"],
            "is_all_selected": true/false
        },
        "other_filters": [{"name": "<filter name>", "selected_value": "<value>"}]
    },
    "charts": [
        {
            "title": "<exact chart title>",
            "type": "<bar/line/pie/donut/map/table/card/gauge/other>",
            "data_points": [
                {"label": "<category/label>", "value": "<exact numeric value>", "percentage": "<if shown>"}
            ],
            "legend_items": ["<legend labels if visible>"],
            "axis_labels": {"x": "<x-axis label>", "y": "<y-axis label>"},
            "total_or_summary": "<any total/summary value shown>"
        }
    ],
    "tables": [
        {
            "title": "<table title>",
            "columns": ["<column headers>"],
            "sample_rows": [["<row data>"]],
            "row_count": "<total rows if visible>"
        }
    ],
    "top_items": {
        "top_airlines_by_flights": [{"airline": "<name>", "flights": "<count>"}],
        "top_airports": [{"airport": "<code/name>", "value": "<metric>"}],
        "delay_causes": [{"cause": "<delay type>", "value": "<time or count>", "percentage": "<if shown>"}],
        "cancellation_reasons": [{"reason": "<type>", "count": "<number>", "percentage": "<if shown>"}]
    },
    "visible_time_period": "<exact date range text shown anywhere on dashboard>",
    "dashboard_title": "<main title if visible>",
    "data_freshness": "<last updated date if shown>",
    "notable_values": ["<any standout metrics, warnings, or highlighted values>"]
}

IMPORTANT: 
- Extract REAL numbers from pie charts (read the percentages/values shown)
- For delay causes: extract each cause and its exact value
- For cancelled flights: extract breakdown by reason if visible
- Look for any date picker or slicer values at the top/side of the dashboard

Analyze the image THOROUGHLY and return ONLY the JSON with maximum detail. No other text."""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": extraction_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.1  # Low temperature for more accurate extraction
    }
    
    try:
        print("🔍 Sending screenshot to Groq Vision API...")
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        print("✅ Groq Vision analysis complete")
        
        # Try to parse as JSON
        try:
            # Find JSON in the response (might be wrapped in markdown code blocks)
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()
            
            extracted_data = json.loads(json_str)
            return {"success": True, "data": extracted_data, "raw_response": content}
        except json.JSONDecodeError as je:
            print(f"⚠️ JSON parse error: {je}")
            # Return raw text if JSON parsing fails
            return {"success": True, "data": None, "raw_response": content}
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Groq API error: {e}")
        error_detail = ""
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
            except:
                error_detail = e.response.text
        return {"success": False, "error": f"{str(e)} - {error_detail}", "data": None}


def format_extracted_data_for_llm(extracted_result: dict) -> str:
    """
    Format the extracted data into a context string for the Azure OpenAI LLM.
    
    Args:
        extracted_result: Result from extract_dashboard_data_with_groq
        
    Returns:
        Formatted string context for LLM
    """
    if not extracted_result.get("success"):
        return f"Error extracting dashboard data: {extracted_result.get('error', 'Unknown error')}"
    
    data = extracted_result.get("data")
    
    if data is None:
        # Use raw response if JSON parsing failed
        raw = extracted_result.get('raw_response', 'No data available')
        return f"""=== DASHBOARD DATA (Raw Extraction) ===
{raw}
========================================"""
    
    # Post-process and normalize some KPI/chart fields to avoid misinterpretation
    def _format_rate_for_display(raw):
        # Return a human-friendly percent string and a numeric percent value when possible
        if raw is None:
            return None, None
        try:
            if isinstance(raw, str) and '%' in raw:
                s = raw.replace('%', '').strip()
                v = float(s)
                return f"{v:.6f}%" if abs(v) < 1.0 else f"{v:.2f}%", v
            n = float(raw)
            # Heuristic: very small decimals (<0.01) are already percents like 0.003198 -> 0.003198%
            if abs(n) < 0.01:
                return f"{n:.6f}%", n
            # If value between 0.01 and 1.0, treat as fraction (0-1) -> percent
            if 0.01 <= abs(n) <= 1.0:
                p = n * 100.0
                return f"{p:.3f}%", p
            # Values > 1.0 likely already in percent units
            return f"{n:.3f}%", n
        except Exception:
            return str(raw), None

    # Normalize kpi rate displays (preserve original values in data)
    if data and isinstance(data, dict) and data.get('kpis'):
        kpis = data['kpis']
        if 'cancellation_rate' in kpis:
            disp, pct = _format_rate_for_display(kpis.get('cancellation_rate'))
            kpis['cancellation_rate_display'] = disp
            kpis['cancellation_rate_pct'] = pct
        if 'on_time_rate' in kpis:
            disp2, pct2 = _format_rate_for_display(kpis.get('on_time_rate'))
            kpis['on_time_rate_display'] = disp2
            kpis['on_time_rate_pct'] = pct2

    # Clean top_items: remove erroneous 'total' rows from cancellation_reasons/delay_causes
    if data and isinstance(data, dict) and data.get('top_items'):
        top = data['top_items']
        # Helper to filter totals
        def _filter_totals(items):
            kept = []
            total_item = None
            for it in items:
                reason = (it.get('reason') or it.get('cause') or '').strip()
                if isinstance(reason, str) and reason.lower() in ('total', 'all', 'overall', 'sum'):
                    total_item = it
                else:
                    kept.append(it)
            return kept, total_item

        if 'cancellation_reasons' in top and isinstance(top['cancellation_reasons'], list):
            filtered, total_it = _filter_totals(top['cancellation_reasons'])
            top['cancellation_reasons'] = filtered
            if total_it:
                top['cancellation_total'] = total_it

        if 'delay_causes' in top and isinstance(top['delay_causes'], list):
            filtered_d, total_d = _filter_totals(top['delay_causes'])
            top['delay_causes'] = filtered_d
            if total_d:
                top['delay_total'] = total_d

    lines = ["=== LIVE DASHBOARD DATA (Extracted via Vision AI) ===", ""]
    
    # Dashboard title and time period
    if data.get("dashboard_title"):
        lines.append(f"📊 Dashboard: {data['dashboard_title']}")
    if data.get("visible_time_period"):
        lines.append(f"📅 Time Period: {data['visible_time_period']}")
    if data.get("data_freshness"):
        lines.append(f"🕐 Data Freshness: {data['data_freshness']}")
    lines.append("")
    
    # Slicers and Filters (NEW)
    if "slicers_and_filters" in data:
        filters = data["slicers_and_filters"]
        lines.append("🔍 ACTIVE FILTERS & SLICERS:")
        
        if filters.get("date_range"):
            dr = filters["date_range"]
            if dr.get("full_range_text"):
                lines.append(f"  • Date Range: {dr['full_range_text']}")
            elif dr.get("start_date") or dr.get("end_date"):
                start = dr.get("start_date", "?")
                end = dr.get("end_date", "?")
                lines.append(f"  • Date Range: {start} to {end}")
        
        if filters.get("airline_filter"):
            af = filters["airline_filter"]
            if af.get("is_all_selected"):
                lines.append("  • Airlines: All Selected")
            elif af.get("selected_airlines"):
                lines.append(f"  • Airlines: {', '.join(af['selected_airlines'])}")
        
        if filters.get("other_filters"):
            for f in filters["other_filters"]:
                lines.append(f"  • {f.get('name', 'Filter')}: {f.get('selected_value', '?')}")
        lines.append("")
    
    # KPIs
    if "kpis" in data:
        lines.append("📊 KEY PERFORMANCE INDICATORS:")
        kpis = data["kpis"]
        standard_kpis = ["total_flights", "on_time_rate", "cancellation_rate", "average_delay", "total_airlines", "total_airports"]
        for key in standard_kpis:
            value = kpis.get(key)
            if value and value != "unclear":
                display_key = key.replace("_", " ").title()
                lines.append(f"  • {display_key}: {value}")
        
        # Other KPIs
        if kpis.get("other_kpis"):
            for kpi in kpis["other_kpis"]:
                lines.append(f"  • {kpi.get('name', 'KPI')}: {kpi.get('value', '?')}")
        lines.append("")

        # (KPI contextual notes for Azure are appended when sending to Azure)
    
    # Charts with actual data
    if "charts" in data and data["charts"]:
        lines.append("📈 CHARTS & VISUALIZATIONS:")
        for chart in data["charts"]:
            title = chart.get('title', 'Untitled')
            chart_type = chart.get('type', 'unknown')
            lines.append(f"  📉 {title} ({chart_type})")
            
            # Show data points
            if chart.get("data_points"):
                lines.append("    Data:")
                for dp in chart["data_points"][:10]:  # Limit to 10 points
                    label = dp.get("label", "?")
                    value = dp.get("value", "?")
                    pct = dp.get("percentage", "")
                    pct_str = f" ({pct})" if pct else ""
                    lines.append(f"      - {label}: {value}{pct_str}")
            
            if chart.get("total_or_summary"):
                lines.append(f"    Total/Summary: {chart['total_or_summary']}")
        lines.append("")
    
    # Tables
    if "tables" in data and data["tables"]:
        lines.append("📋 TABLES:")
        for table in data["tables"]:
            lines.append(f"  • {table.get('title', 'Table')}")
            if table.get("columns"):
                lines.append(f"    Columns: {', '.join(table['columns'])}")
            if table.get("row_count"):
                lines.append(f"    Rows: {table['row_count']}")
        lines.append("")
    
    # Top items with detailed data
    if "top_items" in data:
        top = data["top_items"]
        
        if top.get("top_airlines_by_flights"):
            lines.append("🏆 TOP AIRLINES:")
            for item in top["top_airlines_by_flights"][:5]:
                lines.append(f"  • {item.get('airline', '?')}: {item.get('flights', '?')} flights")
            lines.append("")
        
        if top.get("top_airports"):
            lines.append("🛫 TOP AIRPORTS:")
            for item in top["top_airports"][:5]:
                lines.append(f"  • {item.get('airport', '?')}: {item.get('value', '?')}")
            lines.append("")
        
        if top.get("delay_causes"):
            lines.append("⏰ DELAY CAUSES:")
            for item in top["delay_causes"]:
                cause = item.get("cause", "?")
                value = item.get("value", "?")
                pct = item.get("percentage", "")
                pct_str = f" ({pct})" if pct else ""
                lines.append(f"  • {cause}: {value}{pct_str}")
            lines.append("")
        
        if top.get("cancellation_reasons"):
            lines.append("❌ CANCELLATION REASONS:")
            for item in top["cancellation_reasons"]:
                reason = item.get("reason", "?")
                count = item.get("count", "?")
                pct = item.get("percentage", "")
                pct_str = f" ({pct})" if pct else ""
                lines.append(f"  • {reason}: {count}{pct_str}")
            lines.append("")
    
    # Notable values
    if data.get("notable_values"):
        lines.append("💡 NOTABLE VALUES:")
        for val in data["notable_values"]:
            lines.append(f"  • {val}")
        lines.append("")
    
    lines.append("=" * 55)
    
    # --- Computed Delay Causes Summary (helps LLM reason accurately) ---
    def _parse_number(s: str):
        if s is None:
            return None
        try:
            # Remove commas and unit words, then extract first float
            import re
            s_clean = str(s).replace(',', '')
            m = re.search(r"[-+]?[0-9]*\.?[0-9]+", s_clean)
            return float(m.group(0)) if m else None
        except Exception:
            return None

    computed_lines = []
    if "top_items" in data and data["top_items"].get("delay_causes"):
        causes = data["top_items"]["delay_causes"]
        parsed = []
        # Color mapping for known causes (ensure 'security' shows as yellow)
        color_map = {
            'security': 'yellow'
        }
        for c in causes:
            cause = c.get("cause")
            val_raw = c.get("value")
            pct_raw = c.get("percentage")
            val_num = _parse_number(val_raw)
            pct_num = _parse_number(pct_raw)
            color = None
            try:
                if cause:
                    color = color_map.get(str(cause).strip().lower())
            except Exception:
                color = None
            parsed.append({
                "cause": cause,
                "value_raw": val_raw,
                "pct_raw": pct_raw,
                "value": val_num,
                "pct": pct_num,
                "color": color
            })
        # Remove any items that are clearly 'total' or summary rows which can distort rankings
        try:
            import re
            def _is_total_label(s):
                if not s:
                    return False
                s2 = str(s).strip().lower()
                return bool(re.search(r"\b(total|all|overall|sum|total_delay|combined)\b", s2))

            filtered = [p for p in parsed if not _is_total_label(p.get('cause'))]
            # If filtering removed everything, avoid falling back to the total/summary row
            # (falling back can incorrectly attribute the 'total' value to a specific cause like 'security')
            if filtered:
                parsed = filtered
            else:
                # No reliable per-cause rows found — skip computed summary to avoid misattribution
                parsed = []
        except Exception:
            pass

        # If parsed is empty (no reliable per-cause rows), skip summary
        if parsed:
            # If percentages missing, compute from values (if values available)
            pct_sum = sum([p["pct"] for p in parsed if p["pct"] is not None])
            vals_available = any(p["value"] is not None for p in parsed)

            if (pct_sum == 0 or pct_sum is None) and vals_available:
                total = sum([p["value"] for p in parsed if p["value"] is not None]) or 0
                if total > 0:
                    for p in parsed:
                        if p["value"] is not None:
                            p["computed_pct"] = round((p["value"] / total) * 100, 2)
                        else:
                            p["computed_pct"] = None
            else:
                # Normalize provided percentages to sum to ~100 if they exist
                if pct_sum and abs(pct_sum - 100.0) > 1.0:
                    factor = 100.0 / pct_sum
                    for p in parsed:
                        if p["pct"] is not None:
                            p["computed_pct"] = round(p["pct"] * factor, 2)
                        else:
                            p["computed_pct"] = None
                else:
                    for p in parsed:
                        p["computed_pct"] = p["pct"]

            # Build a short summary
            # Sort by computed_pct (fallback to value)
            ranked = sorted(parsed, key=lambda x: (x.get("computed_pct") if x.get("computed_pct") is not None else (x.get("value") or 0)), reverse=True)
            computed_lines.append("--- COMPUTED DELAY CAUSES SUMMARY ---")
            for p in ranked:
                pct_disp = f"{p['computed_pct']}%" if p.get("computed_pct") is not None else (str(p.get("pct_raw")) if p.get("pct_raw") else "?")
                val_disp = str(p.get("value_raw")) if p.get("value_raw") is not None else "?"
                color_disp = f" [{p.get('color')}]" if p.get('color') else ""
                computed_lines.append(f"  • {p.get('cause', '?')}: {val_disp} — {pct_disp}{color_disp}")

            top_by_pct = ranked[0]
            computed_lines.append(f"Top cause by share: {top_by_pct.get('cause', '?')} ({top_by_pct.get('computed_pct') or top_by_pct.get('pct_raw') or '?' }%)")
            # NOTE: delay cause values are in minutes unless the dashboard specifies otherwise
            computed_lines.append("Note: delay cause values are in minutes unless specified otherwise.")
            lines.append("")
            lines.extend(computed_lines)

    lines.append("=" * 55)

    return "\n".join(lines)


def send_extracted_data_to_azure(extracted_result: dict, question: str, max_tokens: int = 1024, temperature: float = 0.0) -> dict:
    """
    Send the extracted dashboard data to Azure OpenAI (Azure AI) as context and ask a question.

    The LLM is instructed to answer solely based on the provided data. If the answer is
    not present in the provided data, it will reply with 'NOT AVAILABLE IN PROVIDED DATA'.

    Returns a dict with keys: success (bool), answer (str) or error (str).
    """
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_DEPLOYMENT:
        return {"success": False, "error": "Azure OpenAI configuration missing (endpoint/deployment/api key)."}

    context = format_extracted_data_for_llm(extracted_result)
    # Append KPI context from `azure_context.py` so Azure receives focused guidance
    context = append_kpi_context(context, extracted_result)

    system_prompt = get_azure_system_prompt()

    user_prompt = f"{context}\n\nQuestion: {question}\n\nProvide a concise, factual answer based only on the data above."

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    url = f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"

    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        rj = resp.json()
        content = rj["choices"][0]["message"]["content"]
        return {"success": True, "answer": content, "raw_response": rj}
    except requests.exceptions.RequestException as e:
        err = str(e)
        try:
            if e.response is not None:
                err_detail = e.response.json()
                err = f"{err} - {err_detail}"
        except Exception:
            pass
        return {"success": False, "error": err}


if __name__ == "__main__":
    # Test with an existing screenshot
    from screenshot_capture import get_latest_screenshot
    
    latest = get_latest_screenshot()
    if latest:
        print(f"Analyzing: {latest}")
        result = extract_dashboard_data_with_groq(latest)
        print("\n--- Extracted Data ---")
        print(json.dumps(result, indent=2, default=str))
        print("\n--- Formatted for LLM ---")
        print(format_extracted_data_for_llm(result))
    else:
        print("No screenshots found. Run screenshot_capture.py first.")
