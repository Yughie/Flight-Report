"""
Azure-specific prompt/context helpers.

This module centralizes Azure system prompts and KPI contextual notes so
that the Azure-related logic and prompts live in a single place.
"""
from typing import Optional


def get_azure_system_prompt() -> str:
    """Return the system prompt used when calling Azure OpenAI."""
    return (
        "You are an analyst AI. You are given structured dashboard data below. "
        "Answer the user's question using ONLY the provided data. Do NOT use external knowledge. "
        "If the information is not present in the provided data, respond exactly: NOT AVAILABLE IN PROVIDED DATA."
    )


def append_kpi_context(context: str, extracted_result: dict) -> str:
    """Append KPI contextual notes (if applicable) to the provided context string.

    The notes explain how KPI 'Goals' should be interpreted relative to active slicers
    (date range, airline filters, etc.). This keeps Azure-specific guidance in one file.
    """
    def _parse_rate(val):
        """Parse a rate value which may be given as a fraction, a percent string, or a small decimal that should be interpreted as percent.

        Returns (raw_str, percent_value_float)
        """
        try:
            # Keep original string form
            raw = val
            if val is None:
                return (None, None)
            # If it's a string and contains '%', parse as percent
            if isinstance(val, str) and '%' in val:
                s = val.replace('%', '').strip()
                p = float(s)
                return (str(val), p)
            # Try numeric parse
            n = float(val)
            # Heuristic: if value is very small (< 0.01) treat it as already a percent (user convention)
            # e.g., 0.003198 -> 0.003198% (very small) per user input.
            if abs(n) < 0.01:
                return (str(val), n)
            # Otherwise treat as fraction (0-1) and convert to percent
            return (str(val), n * 100.0)
        except Exception:
            return (str(val), None)

    try:
        data = extracted_result.get('data') if isinstance(extracted_result, dict) else None
        kpis = data.get('kpis', {}) if isinstance(data, dict) else {}
        top_items = data.get('top_items', {}) if isinstance(data, dict) else {}

        # Default goal thresholds (can be adjusted later)
        GOALS = {
            'on_time_rate': 0.70,       # 70% on-time target
            'cancellation_rate': 0.02   # 2% cancellation target
        }

        kpi_notes = []
        # Clarify observed vs baseline: displayed KPIs are the latest-day values
        # while goals/averages are computed across the full selected date range.
        kpi_notes.append(
            "Note: KPI values displayed on the dashboard represent the most recent day within the active date range (last-day snapshot). "
            "Goal or average baselines are computed across the entire selected date range. For 'Total Flights', the baseline uses successful (non-cancelled) flights only."
        )
        # Slicer context (date range, airline selector)
        slicers = data.get('slicers_and_filters') if isinstance(data, dict) else {}
        if slicers:
            # Date range
            dr = slicers.get('date_range') or {}
            if dr.get('full_range_text'):
                kpi_notes.append(f"Active slicer - Date Range: {dr.get('full_range_text')}")
            elif dr.get('start_date') or dr.get('end_date'):
                start = dr.get('start_date', '?')
                end = dr.get('end_date', '?')
                kpi_notes.append(f"Active slicer - Date Range: {start} to {end}")
            # Airline selector
            af = slicers.get('airline_filter') or {}
            if af:
                if af.get('is_all_selected'):
                    kpi_notes.append("Active slicer - Airlines: All selected")
                elif af.get('selected_airlines'):
                    kpi_notes.append(f"Active slicer - Airlines: {', '.join(af.get('selected_airlines')[:5])}")
        if kpis.get('total_flights'):
            kpi_notes.append(
                "'Total Flights' Goal: baseline average of successful (non-cancelled) flights. "
                "Affected by active slicers such as date range and selected airlines; changing slicers changes the applicable baseline."
            )

        # On-Time Performance: include observed value and goal, link to delay causes
        if kpis.get('on_time_rate'):
            raw_obs = kpis.get('on_time_rate')
            raw_str, obs_percent = _parse_rate(raw_obs)
            obs_display = f"{obs_percent:.3f}%" if obs_percent is not None else str(raw_obs)
            kpi_notes.append(
                f"'On-Time Performance' Observed (interpreted): {obs_display}. Goal: {GOALS['on_time_rate']*100:.0f}% (i.e. {GOALS['on_time_rate']}). "
                "If observed < goal, attribute shortfall to delay causes in 'top_items.delay_causes' (e.g., departure/arrival/airline/system/late-aircraft)."
            )
            # List top 3 delay causes if available
            delay_causes = top_items.get('delay_causes') or []
            if delay_causes:
                top_delay = [d.get('cause', '?') for d in delay_causes[:3]]
                kpi_notes.append(f"  Top delay contributors (from dashboard): {', '.join(top_delay)}.")

        # Charts: detect delay-related charts (e.g., "Average Delay Causes") and note units
        charts = data.get('charts') if isinstance(data, dict) else None
        if charts:
            for ch in charts:
                title = (ch.get('title') or '').lower()
                if 'delay' in title:
                    # Note that delay charts are in minutes and depend on slicers
                    kpi_notes.append(
                        f"Chart '{ch.get('title', 'Delay Chart')}' shows average delay causes (values in minutes). "
                        "These values depend on active slicers such as date range and selected airlines; if the chart is stacked, values are component minutes per cause."
                    )
                    # List top data point labels if present
                    dps = ch.get('data_points') or []
                    if dps:
                        dp_labels = [dp.get('label', '?') for dp in dps[:5]]
                        kpi_notes.append(f"  Chart top series/labels: {', '.join(dp_labels)}.")
                    break
            # Cancellation charts: show counts and percentages and depend on slicers
            if charts:
                for ch in charts:
                    title = (ch.get('title') or '').lower()
                    if 'cancel' in title or 'cancelled' in title:
                        kpi_notes.append(
                            f"Chart '{ch.get('title', 'Cancelled Flight')}' shows cancelled flight totals and percentages for the selected range. "
                            "Values typically include absolute counts and percent share by reason; these depend on active slicers such as date range and selected airlines."
                        )
                        # List top data point labels if present
                        dps = ch.get('data_points') or []
                        if dps:
                            dp_labels = [dp.get('label', '?') for dp in dps[:5]]
                            kpi_notes.append(f"  Chart breakdown labels: {', '.join(dp_labels)}.")
                        # If top_items has cancellation_reasons, surface them too
                        cancel_reasons = top_items.get('cancellation_reasons') or []
                        if cancel_reasons:
                            top_cancel = [c.get('reason', '?') for c in cancel_reasons[:3]]
                            kpi_notes.append(f"  Top cancellation reasons (from dashboard): {', '.join(top_cancel)}.")
                        break

        # Cancellation Rate: observed vs goal, link to cancellation reasons
        if kpis.get('cancellation_rate'):
            raw_cancel = kpis.get('cancellation_rate')
            raw_str_c, cancel_percent = _parse_rate(raw_cancel)
            # cancel_percent now represents percentage value (e.g., 0.003198 -> 0.003198%) per heuristic
            cancel_display = f"{cancel_percent:.6f}%" if cancel_percent is not None else str(raw_cancel)
            goal_percent = GOALS['cancellation_rate'] * 100.0
            kpi_notes.append(
                f"'Cancellation Rate' Observed (interpreted): {cancel_display}. Goal: {GOALS['cancellation_rate']} ({goal_percent:.2f}%). Lower is better. "
                "If observed > goal, examine 'top_items.cancellation_reasons' and the 'Cancelled Flight' chart breakdown for root causes (e.g., Weather, Carrier, NAS)."
            )
            cancel_reasons = top_items.get('cancellation_reasons') or []
            if cancel_reasons:
                top_cancel = [c.get('reason', '?') for c in cancel_reasons[:3]]
                kpi_notes.append(f"  Top cancellation reasons (from dashboard): {', '.join(top_cancel)}.")

        if kpi_notes:
            appended = context + "\n\n--- KPI CONTEXT (for Azure LLM) ---\n" + "\n".join([f"  • {n}" for n in kpi_notes])
            return appended
    except Exception:
        pass
    return context
