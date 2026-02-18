"""Power BI data fetching via DAX queries (executeQueries REST API).

Fetches the actual data viewers see in the report, parameterised by
the same airline / date-range filters the slicer controls.

Requirements:
  - The workspace must be on Power BI Premium, Premium Per User (PPU),
    or Embedded capacity.
  - The service-principal app registration needs the
    ``Dataset.Read.All`` (or ``Dataset.ReadWrite.All``) API permission.
"""

from __future__ import annotations

import requests
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import WORKSPACE_ID

# Module-level cache for airline list (never changes within a session)
_airline_list_cache: list[str] | None = None


# ── Low-level helpers ─────────────────────────────────────────────


def get_dataset_id(aad_token: str) -> str:
    """Retrieve the dataset ID linked to the configured report."""
    from .config import REPORT_ID

    headers = {"Authorization": f"Bearer {aad_token}"}
    url = (
        f"https://api.powerbi.com/v1.0/myorg/groups/{WORKSPACE_ID}"
        f"/reports/{REPORT_ID}"
    )
    res = requests.get(url, headers=headers, timeout=30)
    if res.status_code != 200:
        raise ConnectionError(
            f"Report lookup failed ({res.status_code}): {res.text}"
        )
    return res.json()["datasetId"]


def execute_dax(aad_token: str, dataset_id: str, dax: str) -> list:
    """Execute a single DAX query and return the result rows."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {aad_token}",
    }
    url = (
        f"https://api.powerbi.com/v1.0/myorg/groups/{WORKSPACE_ID}"
        f"/datasets/{dataset_id}/executeQueries"
    )
    body = {
        "queries": [{"query": dax}],
        "serializerSettings": {"includeNulls": True},
    }

    # Use 90 second timeout for DAX queries (can be slow on large datasets)
    try:
        res = requests.post(url, headers=headers, json=body, timeout=90)
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "DAX query timed out after 90 seconds. This usually means:\n"
            "1. Your workspace doesn't have Premium/PPU capacity (executeQueries requires Premium)\n"
            "2. The dataset is very large and queries are slow\n"
            "3. Network connectivity issues\n"
            "Most likely cause: Missing Premium/PPU capacity."
        )
    
    if res.status_code != 200:
        # Surface a friendly hint when the capacity is not Premium
        try:
            err_msg = res.json().get("error", {}).get("message", "")
        except Exception:
            err_msg = res.text
        if "premium" in err_msg.lower() or "capacity" in err_msg.lower():
            raise RuntimeError(
                "DAX executeQueries requires a Power BI Premium / PPU "
                "workspace. Please check your capacity settings."
            )
        raise RuntimeError(
            f"DAX query failed ({res.status_code}): {err_msg}"
        )
    data = res.json()

    try:
        rows = data["results"][0]["tables"][0]["rows"]

        # Power BI returns column keys in the format  table[Column Name]
        # e.g. "gold_aviation_report[CANCELLATION REASON]"
        # Extract only the part inside the last pair of brackets so all
        # consumers can use the bare column name as the dict key.
        def _clean_key(k: str) -> str:
            # Try to extract the content inside the last [...] pair
            lb = k.rfind("[")
            rb = k.rfind("]")
            if lb != -1 and rb != -1 and rb > lb:
                return k[lb + 1 : rb]
            # Fallback: strip leading/trailing brackets (simple column names)
            return k.strip("[]")

        cleaned_rows = []
        for row in rows:
            cleaned_row = {}
            for key, value in row.items():
                cleaned_row[_clean_key(key)] = value
            cleaned_rows.append(cleaned_row)

        return cleaned_rows
    except (KeyError, IndexError) as e:
        # Log what we got back to help debug
        import json
        error_detail = json.dumps(data, indent=2, default=str)[:500]  # First 500 chars
        raise RuntimeError(
            f"DAX query succeeded but returned unexpected structure.\n"
            f"Error: {e}\n"
            f"Response preview: {error_detail}"
        )


# ── DAX filter builders ──────────────────────────────────────────


def _date_condition(date_from: str | None, date_to: str | None) -> str:
    """Return the boolean condition for a FLIGHT_DATE filter.

    Uses inclusive endpoints (``>=`` / ``<=``) to match the Power BI
    'between' slicer behaviour.
    """
    parts: list[str] = []
    if date_from:
        y, m, d = date_from.split("-")
        parts.append(
            f"'gold_aviation_report'[FLIGHT_DATE] >= DATE({y}, {int(m)}, {int(d)})"
        )
    if date_to:
        y, m, d = date_to.split("-")
        parts.append(
            f"'gold_aviation_report'[FLIGHT_DATE] <= DATE({y}, {int(m)}, {int(d)})"
        )
    if len(parts) == 2:
        return f"AND(\n                {parts[0]},\n                {parts[1]}\n            )"
    return parts[0]


def _build_defines(
    airline: str | None,
    date_from: str | None,
    date_to: str | None,
) -> tuple[str, list[str]]:
    """Build a DEFINE block and the list of filter-variable names.

    Returns
    -------
    define_block : str
        The ``DEFINE\\n  VAR …`` text (empty string when no filters).
    refs : list[str]
        Variable names (e.g. ``["__DateFilter", "__AirlineFilter"]``).
    """
    var_parts: list[str] = []
    refs: list[str] = []

    if date_from or date_to:
        cond = _date_condition(date_from, date_to)
        var_parts.append(
            "    VAR __DateFilter =\n"
            "        FILTER(\n"
            "            KEEPFILTERS(VALUES('gold_aviation_report'[FLIGHT_DATE])),\n"
            f"            {cond}\n"
            "        )"
        )
        refs.append("__DateFilter")

    if airline:
        escaped = airline.replace("'", "''")
        var_parts.append(
            "    VAR __AirlineFilter =\n"
            f'        TREATAS({{"{escaped}"}}, '
            "'gold_aviation_report'[AIRLINE_NAME])"
        )
        refs.append("__AirlineFilter")

    if var_parts:
        define_block = "DEFINE\n" + "\n\n".join(var_parts) + "\n\n"
    else:
        define_block = ""
    return define_block, refs


# ── Individual DAX fetchers ───────────────────────────────────────


def fetch_airline_list(
    token: str, ds_id: str,
) -> list[str]:
    """Return the distinct airline names available in the dataset."""
    query = (
        "EVALUATE\n"
        "    SUMMARIZECOLUMNS(\n"
        "        'gold_aviation_report'[AIRLINE_NAME]\n"
        "    )\n"
        "ORDER BY 'gold_aviation_report'[AIRLINE_NAME]\n"
    )
    rows = execute_dax(token, ds_id, query)
    return [
        r.get("AIRLINE_NAME") or r.get("gold_aviation_report[AIRLINE_NAME]", "")
        for r in rows
        if r.get("AIRLINE_NAME") or r.get("gold_aviation_report[AIRLINE_NAME]")
    ]


def fetch_date_range(
    token: str, ds_id: str,
    airline=None, date_from=None, date_to=None,
) -> dict:
    """Min / max flight date + totals for the active filter.

    Also returns last-day stats:
        LastFlightDate, LastDayFlights, LastDayCancelled,
        LastDayCancellationRate, LastDayOTP
    """
    # Build the standard date/airline filter VARs via _build_defines,
    # then append a __LastDate VAR so we can cheaply reference it without
    # nesting CALCULATE(MAX(...)) inside a filter predicate (which causes
    # DAX 400 errors in ROW() context).
    _, refs = _build_defines(airline, date_from, date_to)
    calc_args = (", " + ", ".join(refs)) if refs else ""

    # Collect all VAR lines manually so __LastDate can depend on the others
    var_parts: list[str] = []

    if date_from or date_to:
        cond = _date_condition(date_from, date_to)
        var_parts.append(
            "    VAR __DateFilter =\n"
            "        FILTER(\n"
            "            KEEPFILTERS(VALUES('gold_aviation_report'[FLIGHT_DATE])),\n"
            f"            {cond}\n"
            "        )"
        )

    if airline:
        escaped = airline.replace("'", "''")
        var_parts.append(
            "    VAR __AirlineFilter =\n"
            f'        TREATAS({{"{escaped}"}}, '
            "'gold_aviation_report'[AIRLINE_NAME])"
        )

    # __LastDate captures the most-recent date respecting active filters
    var_parts.append(
        f"    VAR __LastDate = CALCULATE(MAX('gold_aviation_report'[FLIGHT_DATE]){calc_args})"
    )

    define_block = "DEFINE\n" + "\n\n".join(var_parts) + "\n\n"

    query = (
        f"{define_block}"
        "EVALUATE\n"
        "    ROW(\n"
        f'        "MinFLIGHT_DATE", CALCULATE(MIN(\'gold_aviation_report\'[FLIGHT_DATE]){calc_args}),\n'
        f'        "MaxFLIGHT_DATE", CALCULATE(MAX(\'gold_aviation_report\'[FLIGHT_DATE]){calc_args}),\n'
        f'        "TotalFlights", CALCULATE(COUNTROWS(\'gold_aviation_report\'){calc_args}),\n'
        f'        "TotalCancelled", CALCULATE(SUM(\'gold_aviation_report\'[CANCELLED]){calc_args}),\n'
        f'        "LastFlightDate", __LastDate,\n'
        f'        "LastDayFlights", CALCULATE(COUNTROWS(\'gold_aviation_report\'), \'gold_aviation_report\'[FLIGHT_DATE] = __LastDate{calc_args}),\n'
        f'        "LastDayCancelled", CALCULATE(SUM(\'gold_aviation_report\'[CANCELLED]), \'gold_aviation_report\'[FLIGHT_DATE] = __LastDate{calc_args}),\n'
        f'        "LastDayCancellationRate", DIVIDE(\n'
        f'            CALCULATE(SUM(\'gold_aviation_report\'[CANCELLED]), \'gold_aviation_report\'[FLIGHT_DATE] = __LastDate{calc_args}),\n'
        f'            CALCULATE(COUNTROWS(\'gold_aviation_report\'), \'gold_aviation_report\'[FLIGHT_DATE] = __LastDate{calc_args}), 0),\n'
        f'        "LastDayOTP", DIVIDE(\n'
        f'            CALCULATE(COUNTROWS(\'gold_aviation_report\'), \'gold_aviation_report\'[FLIGHT_DATE] = __LastDate, \'gold_aviation_report\'[ARRIVAL_DELAY] <= 15{calc_args}),\n'
        f'            CALCULATE(COUNTROWS(\'gold_aviation_report\'), \'gold_aviation_report\'[FLIGHT_DATE] = __LastDate{calc_args}), 0)\n'
        "    )\n"
    )
    rows = execute_dax(token, ds_id, query)
    return rows[0] if rows else {}


def fetch_delay_summary(
    token: str, ds_id: str,
    airline=None, date_from=None, date_to=None,
) -> dict:
    """Sum of each delay-type column."""
    # Use DEFINE + CALCULATE(<expr>, <filter refs>) to ensure the
    # same dynamic date/airline filters are applied consistently to
    # each numerator/denominator. Returns a single-row dict of sums.
    define, refs = _build_defines(airline, date_from, date_to)
    calc_args = (", " + ", ".join(refs)) if refs else ""
    query = (
        f"{define}"
        "EVALUATE\n"
        "    ROW(\n"
        f'        "SumDEPARTURE_DELAY", CALCULATE(SUM(\'gold_aviation_report\'[DEPARTURE_DELAY]){calc_args}),\n'
        f'        "SumLATE_AIRCRAFT_DELAY", CALCULATE(SUM(\'gold_aviation_report\'[LATE_AIRCRAFT_DELAY]){calc_args}),\n'
        f'        "SumAIRLINE_DELAY", CALCULATE(SUM(\'gold_aviation_report\'[AIRLINE_DELAY]){calc_args}),\n'
        f'        "SumARRIVAL_DELAY", CALCULATE(SUM(\'gold_aviation_report\'[ARRIVAL_DELAY]){calc_args}),\n'
        f'        "SumSECURITY_DELAY", CALCULATE(SUM(\'gold_aviation_report\'[SECURITY_DELAY]){calc_args}),\n'
        f'        "SumWEATHER_DELAY", CALCULATE(SUM(\'gold_aviation_report\'[WEATHER_DELAY]){calc_args}),\n'
        f'        "SumAIR_SYSTEM_DELAY", CALCULATE(SUM(\'gold_aviation_report\'[AIR_SYSTEM_DELAY]){calc_args})\n'
        "    )\n"
    )
    rows = execute_dax(token, ds_id, query)
    return rows[0] if rows else {}


_CANCEL_CODE_LABELS = {
    "A": "Airline/Carrier",
    "B": "Weather",
    "C": "National Air System",
    "D": "Security",
}


def fetch_cancellation_breakdown(
    token: str, ds_id: str,
    airline=None, date_from=None, date_to=None,
) -> list:
    """Cancellation count per reason.

    Queries the raw 'flights'[CANCELLATION_REASON] codes (A/B/C/D)
    directly from the flights table and sums 'flights'[CANCELLED].
    Date and airline filters are applied via the gold_aviation_report
    table using TREATAS so the model relationships are respected.
    Labels are mapped to readable names in Python after the query.
    """
    var_parts: list[str] = []
    sc_refs: list[str] = []

    # Only include the four real cancellation codes — exclude "Not Cancelled" / blanks
    var_parts.append(
        '    VAR __CancelCodesFilter =\n'
        '        FILTER(\n'
        '            KEEPFILTERS(VALUES(\'flights\'[CANCELLATION_REASON])),\n'
        '            \'flights\'[CANCELLATION_REASON] IN {"A", "B", "C", "D"}\n'
        '        )'
    )
    sc_refs.append("__CancelCodesFilter")

    if date_from or date_to:
        cond = _date_condition(date_from, date_to)
        var_parts.append(
            "    VAR __DateFilter =\n"
            "        FILTER(\n"
            "            KEEPFILTERS(VALUES('gold_aviation_report'[FLIGHT_DATE])),\n"
            f"            {cond}\n"
            "        )"
        )
        sc_refs.append("__DateFilter")

    if airline:
        escaped = airline.replace("'", "''")
        var_parts.append(
            "    VAR __AirlineFilter =\n"
            f'        TREATAS({{"{escaped}"}}, '
            "'gold_aviation_report'[AIRLINE_NAME])"
        )
        sc_refs.append("__AirlineFilter")

    sc_filter_str = ",\n            ".join(sc_refs)

    var_parts.append(
        "    VAR __Core =\n"
        "        SUMMARIZECOLUMNS(\n"
        "            'flights'[CANCELLATION_REASON],\n"
        f"            {sc_filter_str},\n"
        '            "SumCANCELLED", SUM(\'flights\'[CANCELLED])\n'
        "        )\n"
        "\n"
        "    VAR __Limited =\n"
        "        TOPN(1002, __Core, [SumCANCELLED], 0, 'flights'[CANCELLATION_REASON], 1)"
    )

    define = "DEFINE\n" + "\n\n".join(var_parts) + "\n\n"
    query = (
        f"{define}"
        "EVALUATE __Limited\n"
        "ORDER BY [SumCANCELLED] DESC, 'flights'[CANCELLATION_REASON]\n"
    )
    raw_rows = execute_dax(token, ds_id, query)

    # Map raw letter codes to readable labels so downstream consumers
    # can use row["CANCELLATION REASON"] as before.
    result = []
    for row in raw_rows:
        code = row.get("CANCELLATION_REASON") or ""
        label = _CANCEL_CODE_LABELS.get(code.upper(), code or "Unknown")
        result.append({
            "CANCELLATION REASON": label,
            "SumCANCELLED": row.get("SumCANCELLED", 0) or 0,
        })
    return result


def _fetch_trend(
    token: str, ds_id: str,
    measures_block: str,
    airline=None, date_from=None, date_to=None,
) -> list:
    """Shared helper for daily-trend queries (flight vol / OTP / cancel rate)."""
    # Build filter VARs
    var_parts: list[str] = []
    sc_refs: list[str] = []

    if date_from or date_to:
        cond = _date_condition(date_from, date_to)
        var_parts.append(
            "    VAR __DateFilter =\n"
            "        FILTER(\n"
            "            KEEPFILTERS(VALUES('gold_aviation_report'[FLIGHT_DATE])),\n"
            f"            {cond}\n"
            "        )"
        )
        sc_refs.append("__DateFilter")

    if airline:
        escaped = airline.replace("'", "''")
        var_parts.append(
            "    VAR __AirlineFilter =\n"
            f'        TREATAS({{"{escaped}"}}, '
            "'gold_aviation_report'[AIRLINE_NAME])"
        )
        sc_refs.append("__AirlineFilter")

    sc_filter_line = ""
    if sc_refs:
        sc_filter_line = "            " + ",\n            ".join(sc_refs) + ",\n"

    var_parts.append(
        "    VAR __Core =\n"
        "        SUMMARIZECOLUMNS(\n"
        "            'gold_aviation_report'[FLIGHT_DATE],\n"
        f"{sc_filter_line}"
        f"            {measures_block}\n"
        "        )\n"
        "\n"
        "    VAR __Limited =\n"
        "        TOPN(1002, __Core, 'gold_aviation_report'[FLIGHT_DATE], 0)"
    )

    define = "DEFINE\n" + "\n\n".join(var_parts) + "\n\n"
    query = (
        f"{define}"
        "EVALUATE __Limited\n"
        "ORDER BY 'gold_aviation_report'[FLIGHT_DATE]\n"
    )
    return execute_dax(token, ds_id, query)


def fetch_flight_volume(token, ds_id, airline=None, date_from=None, date_to=None):
    """Actual flights vs goal, per day."""
    return _fetch_trend(
        token, ds_id,
        "\"Actual_Flights\", 'Key Measures'[Actual Flights],\n"
        "            \"Flight_Volume_Goal\", 'Key Measures'[Flight Volume Goal]",
        airline, date_from, date_to,
    )


def fetch_otp(token, ds_id, airline=None, date_from=None, date_to=None):
    """On-time performance % vs target, per day."""
    # Inline the on-time performance calculation to ensure the query
    # remains compatible with dynamic DEFINE filters (date / airline).
    measures_block = (
        '"On_Time_Performance", DIVIDE('
        "CALCULATE(COUNTROWS('gold_aviation_report'), 'gold_aviation_report'[ARRIVAL_DELAY] <= 15), "
        "CALCULATE(COUNTROWS('gold_aviation_report')), 0),\n"
        "            \"Target_OTP\", 'Key Measures'[Target OTP]"
    )
    return _fetch_trend(
        token, ds_id,
        measures_block,
        airline, date_from, date_to,
    )


def fetch_cancellation_rate(token, ds_id, airline=None, date_from=None, date_to=None):
    """Cancellation rate vs target, per day."""
    # Inline the cancellation-rate calculation so DEFINE filters (date/airline)
    # properly apply to both numerator and denominator.
    measures_block = (
        '"Cancellation_Rate", DIVIDE('
        "CALCULATE(SUM('gold_aviation_report'[CANCELLED])), "
        "CALCULATE(COUNTROWS('gold_aviation_report')), 0),\n"
        "            \"Target_Cancellation\", 'Key Measures'[Target cancellation]"
    )
    return _fetch_trend(
        token, ds_id,
        measures_block,
        airline, date_from, date_to,
    )


# ── Orchestrator ──────────────────────────────────────────────────


def fetch_all_report_data(
    aad_token: str,
    dataset_id: str,
    airline: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict:
    """Fetch every KPI / chart dataset for the current slicer state.

    All queries run **in parallel** via a thread pool so the total wait
    time is roughly equal to the *slowest* single query instead of the
    sum of all of them.

    Returns a dict with keys:
        airline_list, date_range, delay_summary, cancellation_breakdown,
        flight_volume, otp, cancellation_rate, errors
    """
    global _airline_list_cache

    data: dict = {}
    errors: list[str] = []

    # Airline list is static — use cached value when available
    if _airline_list_cache is not None:
        data["airline_list"] = _airline_list_cache
    # else: will be fetched in the parallel batch below

    # All filter-dependent queries
    fetchers: list[tuple[str, callable]] = [
        ("date_range", fetch_date_range),
        ("delay_summary", fetch_delay_summary),
        ("cancellation_breakdown", fetch_cancellation_breakdown),
        ("flight_volume", fetch_flight_volume),
        ("otp", fetch_otp),
        ("cancellation_rate", fetch_cancellation_rate),
    ]

    # Include airline_list in the parallel batch when not cached
    include_airline_fetch = _airline_list_cache is None

    with ThreadPoolExecutor(max_workers=7) as pool:
        future_map: dict = {}

        # Submit filter-dependent fetchers
        for key, fn in fetchers:
            future = pool.submit(fn, aad_token, dataset_id, airline, date_from, date_to)
            future_map[future] = key

        # Submit airline list fetch (no filter args)
        if include_airline_fetch:
            future_al = pool.submit(fetch_airline_list, aad_token, dataset_id)
            future_map[future_al] = "airline_list"

        # Collect results as they complete
        for future in as_completed(future_map):
            key = future_map[future]
            try:
                result = future.result()
                data[key] = result
                # Populate cache for airline list
                if key == "airline_list":
                    _airline_list_cache = result
            except Exception as exc:
                errors.append(f"{key}: {exc}")
                if key == "airline_list":
                    data[key] = []
                elif key in ("date_range", "delay_summary"):
                    data[key] = {}
                else:
                    data[key] = []

    if errors:
        data["errors"] = errors
    return data


# ── Formatting for AI context ─────────────────────────────────────

# KPI thresholds / goals (aligned with Power BI report targets)
KPI_GOALS = {
    "otp_target": 0.80,             # 80 % on-time performance target
    "cancellation_target": 0.02,    # 2 % cancellation rate target
    "otp_critical": 0.70,           # below 70 % → critical
    "cancellation_critical": 0.05,  # above 5 % → critical
}


def _fmt(val) -> str:
    """Human-friendly number formatting."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{int(val):,}" if val == int(val) else f"{val:,.1f}"
    if isinstance(val, int):
        return f"{val:,}"
    return str(val)


def _aggregate_monthly(
    rows: list[dict],
    date_key: str,
    measures: dict[str, str],
) -> dict[str, dict]:
    """Collapse daily rows into monthly aggregates (sum or avg)."""
    buckets: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        # Try to get date value with or without table prefix
        raw = (
            row.get(date_key) or
            row.get(f"gold_aviation_report[{date_key}]") or
            row.get(f"[gold_aviation_report[{date_key}]]") or
            ""
        )
        if not raw:
            continue
        month = str(raw)[:7]  # "2015-01"
        for measure in measures:
            val = row.get(measure)
            if val is not None:
                buckets[month][measure].append(val)

    result: dict[str, dict] = {}
    for month, measure_vals in buckets.items():
        result[month] = {}
        for measure, agg in measures.items():
            vals = measure_vals.get(measure, [])
            if not vals:
                result[month][measure] = 0
            elif agg == "sum":
                result[month][measure] = sum(vals)
            elif agg == "avg":
                result[month][measure] = sum(vals) / len(vals)
            else:
                result[month][measure] = sum(vals)
    return result


MONTH_LABELS = {
    "01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr",
    "05": "May", "06": "Jun", "07": "Jul", "08": "Aug",
    "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec",
}


# ── Health assessment helpers ─────────────────────────────────────


def _health_badge(status: str) -> str:
    """Return a text badge for health status."""
    return {
        "critical": "🔴 CRITICAL",
        "warning": "🟡 WARNING",
        "good": "🟢 HEALTHY",
        "info": "🔵 INFO",
    }.get(status, "⚪ UNKNOWN")


def _assess_otp(avg_otp: float | None) -> tuple[str, str]:
    """Return (status, description) for on-time performance."""
    if avg_otp is None:
        return ("info", "No OTP data available")
    if avg_otp < KPI_GOALS["otp_critical"]:
        gap = KPI_GOALS["otp_target"] - avg_otp
        return ("critical", f"OTP at {avg_otp:.1%} — {gap:.1%} below target ({KPI_GOALS['otp_target']:.0%}). Severe delays impacting operations.")
    if avg_otp < KPI_GOALS["otp_target"]:
        gap = KPI_GOALS["otp_target"] - avg_otp
        return ("warning", f"OTP at {avg_otp:.1%} — {gap:.1%} below target ({KPI_GOALS['otp_target']:.0%}). Improvement needed.")
    return ("good", f"OTP at {avg_otp:.1%} — meets or exceeds target ({KPI_GOALS['otp_target']:.0%}).")


def _assess_cancellation(rate: float | None) -> tuple[str, str]:
    """Return (status, description) for cancellation rate."""
    if rate is None:
        return ("info", "No cancellation rate data available")
    if rate > KPI_GOALS["cancellation_critical"]:
        excess = rate - KPI_GOALS["cancellation_target"]
        return ("critical", f"Cancellation rate at {rate:.2%} — {excess:.2%} above target ({KPI_GOALS['cancellation_target']:.0%}). Unacceptable level.")
    if rate > KPI_GOALS["cancellation_target"]:
        excess = rate - KPI_GOALS["cancellation_target"]
        return ("warning", f"Cancellation rate at {rate:.2%} — {excess:.2%} above target ({KPI_GOALS['cancellation_target']:.0%}). Needs attention.")
    return ("good", f"Cancellation rate at {rate:.2%} — within target ({KPI_GOALS['cancellation_target']:.0%}).")


def _assess_flight_volume(monthly: dict) -> tuple[str, str]:
    """Return (status, description) by comparing actual vs goal across months."""
    if not monthly:
        return ("info", "No flight volume data available")
    months_below = 0
    total_gap = 0.0
    for m, vals in monthly.items():
        actual = vals.get("Actual_Flights", 0) or 0
        goal = vals.get("Flight_Volume_Goal", 0) or 0
        if goal > 0 and actual < goal:
            months_below += 1
            total_gap += (goal - actual)
    total_months = len(monthly)
    if months_below == 0:
        return ("good", f"Flight volume met or exceeded goal in all {total_months} months.")
    pct = months_below / total_months
    if pct > 0.5:
        return ("warning", f"Flight volume below goal in {months_below}/{total_months} months (total shortfall: {_fmt(total_gap)} flights).")
    return ("good", f"Flight volume below goal in only {months_below}/{total_months} months.")


def _detect_trend(monthly: dict, measure_key: str) -> str:
    """Detect if a monthly metric is trending up, down, or stable."""
    if not monthly or len(monthly) < 2:
        return "insufficient data"
    sorted_months = sorted(monthly.keys())
    vals = [monthly[m].get(measure_key, 0) or 0 for m in sorted_months]
    if len(vals) < 2:
        return "insufficient data"
    # Compare last third vs first third
    n = len(vals)
    third = max(1, n // 3)
    first_third = vals[:third]
    last_third = vals[n - third:] if n > third else vals[-1:]
    avg_first = sum(first_third) / len(first_third)
    avg_last = sum(last_third) / len(last_third)
    if avg_first == 0:
        return "no baseline"
    change_pct = (avg_last - avg_first) / abs(avg_first)
    if change_pct > 0.05:
        return f"↑ trending UP (+{change_pct:.1%} late vs early period)"
    if change_pct < -0.05:
        return f"↓ trending DOWN ({change_pct:.1%} late vs early period)"
    return "→ stable (no significant change)"


def _identify_worst_best_months(monthly: dict, measure_key: str, higher_is_better: bool = True) -> str:
    """Return the best and worst performing months for a metric."""
    if not monthly:
        return ""
    sorted_items = sorted(monthly.items(), key=lambda x: x[1].get(measure_key, 0) or 0, reverse=higher_is_better)
    best_m = sorted_items[0][0]
    worst_m = sorted_items[-1][0]
    best_val = sorted_items[0][1].get(measure_key, 0)
    worst_val = sorted_items[-1][1].get(measure_key, 0)
    best_label = MONTH_LABELS.get(best_m[-2:], best_m)
    worst_label = MONTH_LABELS.get(worst_m[-2:], worst_m)
    if higher_is_better:
        if isinstance(best_val, float) and best_val < 1:
            return f"Best: {best_m} ({best_label}) at {best_val:.1%} | Worst: {worst_m} ({worst_label}) at {worst_val:.1%}"
        return f"Best: {best_m} ({best_label}) at {_fmt(best_val)} | Worst: {worst_m} ({worst_label}) at {_fmt(worst_val)}"
    else:
        if isinstance(worst_val, float) and worst_val < 1:
            return f"Best: {worst_m} ({worst_label}) at {worst_val:.2%} | Worst: {best_m} ({best_label}) at {best_val:.2%}"
        return f"Best: {worst_m} ({worst_label}) at {_fmt(worst_val)} | Worst: {best_m} ({best_label}) at {_fmt(best_val)}"


# ── SOP Recommendations ──────────────────────────────────────────


def _build_sop_recommendations(
    data: dict,
    otp_monthly: dict | None = None,
    cancel_monthly: dict | None = None,
    vol_monthly: dict | None = None,
) -> list[str]:
    """Generate actionable SOP recommendations based on KPI health.

    Mirrors the approach from azure_context.append_kpi_context() but
    applied to the DAX data pipeline with concrete operational guidance.
    """
    sops: list[str] = []
    dr = data.get("date_range") or {}
    ds = data.get("delay_summary") or {}
    cb = data.get("cancellation_breakdown") or []
    total_flights = dr.get("TotalFlights") or 0
    total_cancelled = dr.get("TotalCancelled") or 0
    cancel_rate = (total_cancelled / total_flights) if total_flights > 0 else None

    # ── 1. Cancellation Rate SOP ──
    if cancel_rate is not None and cancel_rate > KPI_GOALS["cancellation_target"]:
        sops.append("─── SOP: HIGH CANCELLATION RATE ───")
        sops.append(f"  Observed: {cancel_rate:.2%} vs Target: {KPI_GOALS['cancellation_target']:.0%}")

        # Identify dominant cancellation reason
        if cb:
            total_cancel_sum = sum(r.get("SumCANCELLED", 0) or 0 for r in cb)
            for row in cb:
                reason = row.get("CANCELLATION REASON") or "Unknown"
                count = row.get("SumCANCELLED", 0) or 0
                share = (count / total_cancel_sum * 100) if total_cancel_sum > 0 else 0
                reason_lower = reason.lower()

                if share < 10:
                    continue  # skip minor contributors

                sops.append(f"  ▸ {reason}: {_fmt(count)} flights ({share:.1f}% of cancellations)")

                if "weather" in reason_lower:
                    sops.append("    → SOP-W1: Enhance weather forecasting integration with ops planning (48h / 24h / 6h horizons)")
                    sops.append("    → SOP-W2: Pre-position spare aircraft at weather-vulnerable hubs")
                    sops.append("    → SOP-W3: Establish proactive rebooking protocols when weather advisories issued")
                    sops.append("    → SOP-W4: Review de-icing capacity and winter operations readiness")
                elif "carrier" in reason_lower or "airline" in reason_lower:
                    sops.append("    → SOP-C1: Audit maintenance scheduling — identify pattern of MEL deferrals causing cancellations")
                    sops.append("    → SOP-C2: Review crew reserve ratios vs. flight schedule density")
                    sops.append("    → SOP-C3: Implement crew fatigue risk management and scheduling buffers")
                    sops.append("    → SOP-C4: Escalate recurring aircraft-type issues to fleet engineering")
                elif "national air system" in reason_lower or "nas" in reason_lower:
                    sops.append("    → SOP-N1: Coordinate with ATC for slot optimization at congested airports")
                    sops.append("    → SOP-N2: Review alternate routing options for high-congestion corridors")
                    sops.append("    → SOP-N3: Engage in FAA CDM (Collaborative Decision Making) programmes")
                elif "security" in reason_lower:
                    sops.append("    → SOP-S1: Liaise with TSA on checkpoint throughput improvements")
                    sops.append("    → SOP-S2: Review gate-hold procedures during security incidents")
                    sops.append("    → SOP-S3: Assess whether security-driven cancellations correlate with specific airports")
        sops.append("")

    # ── 2. On-Time Performance SOP ──
    if otp_monthly:
        avg_otp_vals = [v.get("On_Time_Performance", 0) or 0 for v in otp_monthly.values()]
        avg_otp = sum(avg_otp_vals) / len(avg_otp_vals) if avg_otp_vals else None
    else:
        avg_otp = None

    if avg_otp is not None and avg_otp < KPI_GOALS["otp_target"]:
        sops.append("─── SOP: LOW ON-TIME PERFORMANCE ───")
        sops.append(f"  Observed avg: {avg_otp:.1%} vs Target: {KPI_GOALS['otp_target']:.0%}")

        # Break down by delay type to give targeted SOPs
        delay_types = {
            "SumDEPARTURE_DELAY": ("Departure Delay", [
                "→ SOP-D1: Review turnaround time standards — identify stations with chronic late departures",
                "→ SOP-D2: Optimize boarding process (zone boarding, early boarding for families/assist)",
                "→ SOP-D3: Reduce ground time gaps — pre-stage catering, cleaning, fueling",
                "→ SOP-D4: Implement gate conflict resolution at hub airports",
            ]),
            "SumLATE_AIRCRAFT_DELAY": ("Late Aircraft Delay", [
                "→ SOP-LA1: Add schedule buffer (padding) on high-utilisation aircraft rotations",
                "→ SOP-LA2: Identify aircraft 'chains' where a single late arrival cascades into 3+ delays",
                "→ SOP-LA3: Strategic spare aircraft placement at top-10 busiest stations",
                "→ SOP-LA4: Monitor aircraft swap effectiveness — track if swaps resolve or shift delays",
            ]),
            "SumAIRLINE_DELAY": ("Airline Delay", [
                "→ SOP-AL1: Root-cause analysis on maintenance-induced delays — scheduled vs unscheduled",
                "→ SOP-AL2: Review crew scheduling rules — minimum rest gaps, deadheading efficiency",
                "→ SOP-AL3: Assess baggage handling and ramp operation bottlenecks",
                "→ SOP-AL4: Weekly ops review meetings with station managers for top-offending stations",
            ]),
            "SumWEATHER_DELAY": ("Weather Delay", [
                "→ SOP-WD1: Cross-reference delays with weather severity — identify over-cautious ground stops",
                "→ SOP-WD2: Diversify hub operations to reduce dependency on weather-prone airports",
                "→ SOP-WD3: Seasonal readiness checklist (winter de-icing, summer thunderstorm protocols)",
            ]),
            "SumAIR_SYSTEM_DELAY": ("Air System / ATC Delay", [
                "→ SOP-AS1: Participate in FAA Traffic Flow Management (TFM) initiatives",
                "→ SOP-AS2: Evaluate RNAV/RNP approach capability to reduce ATC-dependent sequencing",
                "→ SOP-AS3: Lobby for NextGen airspace modernisation at chronic-delay airports",
            ]),
            "SumSECURITY_DELAY": ("Security Delay", [
                "→ SOP-SD1: Track security delay frequency by airport — escalate outliers to TSA liaison",
                "→ SOP-SD2: Review if pre-check / CLEAR adoption rates can reduce screening delays",
            ]),
        }

        if ds:
            # Rank delay types by magnitude
            delay_ranked = sorted(
                [(k, ds.get(k, 0) or 0) for k in delay_types],
                key=lambda x: x[1],
                reverse=True,
            )
            total_delay = sum(v for _, v in delay_ranked) or 1
            for key, minutes in delay_ranked:
                share = minutes / total_delay * 100
                if share < 5:
                    continue
                label, actions = delay_types[key]
                sops.append(f"  ▸ {label}: {_fmt(minutes)} min ({share:.1f}% of total delay)")
                for a in actions:
                    sops.append(f"    {a}")
        sops.append("")

    # ── 3. Flight Volume SOP ──
    if vol_monthly:
        months_below = []
        for m in sorted(vol_monthly):
            actual = vol_monthly[m].get("Actual_Flights", 0) or 0
            goal = vol_monthly[m].get("Flight_Volume_Goal", 0) or 0
            if goal > 0 and actual < goal * 0.95:  # more than 5% shortfall
                gap = goal - actual
                months_below.append((m, actual, goal, gap))
        if months_below:
            sops.append("─── SOP: FLIGHT VOLUME BELOW GOAL ───")
            for m, actual, goal, gap in months_below:
                label = MONTH_LABELS.get(m[-2:], m)
                sops.append(f"  ▸ {m} ({label}): Actual={_fmt(actual)} vs Goal={_fmt(goal)} (shortfall: {_fmt(gap)})")
            sops.append("  → SOP-FV1: Review demand forecasting accuracy — compare forecast vs actual bookings")
            sops.append("  → SOP-FV2: Evaluate route profitability — assess if underperforming routes need schedule adjustments")
            sops.append("  → SOP-FV3: Analyse competitive landscape — check if competitors added capacity on same routes")
            sops.append("  → SOP-FV4: Review cancellation contribution — some shortfall may be from cancelled flights (see cancellation SOPs)")
            sops.append("")

    # ── 4. Delay Dominance SOP (single cause > 40%) ──
    if ds:
        controllable_delays = {
            "SumLATE_AIRCRAFT_DELAY": "Late Aircraft",
            "SumAIRLINE_DELAY": "Airline/Carrier",
            "SumWEATHER_DELAY": "Weather",
            "SumAIR_SYSTEM_DELAY": "Air System/ATC",
            "SumSECURITY_DELAY": "Security",
        }
        total_ctrl = sum(ds.get(k, 0) or 0 for k in controllable_delays)
        if total_ctrl > 0:
            for key, label in controllable_delays.items():
                val = ds.get(key, 0) or 0
                share = val / total_ctrl * 100
                if share > 40:
                    sops.append(f"─── SOP: DOMINANT DELAY TYPE — {label.upper()} ({share:.0f}% of delays) ───")
                    sops.append(f"  This single cause accounts for >{share:.0f}% of total delay minutes.")
                    sops.append("  → Concentrate mitigation resources on this category (see specific SOPs above).")
                    sops.append("  → Escalate to VP Operations for executive-level action plan within 48 hours.")
                    sops.append("  → Establish a dedicated task force or war-room if the trend continues > 2 months.")
                    sops.append("")

    return sops


# ── Main formatting function ─────────────────────────────────────


def format_data_for_ai(
    data: dict,
    airline: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> str:
    """Create a rich, analysis-ready summary of report data for the AI prompt.

    Includes:
      - Raw KPI data (flights, delays, cancellations, OTP)
      - Computed analytics (ratios, averages per flight, shares)
      - Monthly trends with direction indicators
      - Health assessment badges per KPI
      - SOP recommendations when KPIs are below target
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  POWER BI FLIGHT REPORT — DATA & ANALYTICS CONTEXT")
    lines.append("=" * 60)

    # ── ERRORS FIRST (if any) ───
    if data.get("errors"):
        lines.append("")
        lines.append("⚠⚠⚠ DATA FETCH ERRORS ⚠⚠⚠")
        for err in data["errors"]:
            lines.append(f"  ✗ {err}")
        lines.append("")
        lines.append("NOTE: Some or all queries failed. Data below may be incomplete.")
        lines.append("=" * 50)
        lines.append("")

    # ── Available airlines (from dataset)
    airline_list = data.get("airline_list") or []
    if airline_list:
        lines.append(f"Available Airlines ({len(airline_list)}): " + ", ".join(airline_list))
    lines.append("")

    # ── Active filters
    filters: list[str] = []
    if airline:
        filters.append(f"Airline: {airline}")
    if date_from or date_to:
        filters.append(
            f"Date Range: {date_from or '(start)'} to {date_to or '(end)'}"
        )
    lines.append(
        "Active Filters: " + (", ".join(filters) if filters else "None (all airlines, full year)")
    )
    lines.append("")

    # ── Date range / totals + computed metrics ────────────────────
    dr = data.get("date_range") or {}
    total_flights = dr.get("TotalFlights") or 0
    total_cancelled = dr.get("TotalCancelled") or 0
    cancel_rate = (total_cancelled / total_flights) if total_flights > 0 else None
    successful_flights = total_flights - total_cancelled

    if dr:
        lines.append("━━━ HEADLINE METRICS ━━━")
        lines.append(
            f"Date span: {dr.get('MinFLIGHT_DATE', 'N/A')} → {dr.get('MaxFLIGHT_DATE', 'N/A')}"
        )
        lines.append(f"Total Flights: {_fmt(total_flights)}")
        lines.append(f"Successful Flights (non-cancelled): {_fmt(successful_flights)}")
        lines.append(f"Total Cancelled: {_fmt(total_cancelled)}")
        if cancel_rate is not None:
            c_status, c_desc = _assess_cancellation(cancel_rate)
            lines.append(f"Overall Cancellation Rate: {cancel_rate:.2%}  {_health_badge(c_status)}")
            lines.append(f"  Assessment: {c_desc}")

        # Last-day cancellation rate + OTP
        last_date = dr.get("LastFlightDate")
        last_day_flights = dr.get("LastDayFlights") or 0
        last_day_cancelled = dr.get("LastDayCancelled") or 0
        last_day_cr = dr.get("LastDayCancellationRate")
        last_day_otp = dr.get("LastDayOTP")
        if last_date:
            lines.append(f"Last Day ({last_date}): {_fmt(last_day_flights)} flights, {_fmt(last_day_cancelled)} cancelled")
            if last_day_cr is not None:
                ld_cr_status, ld_cr_desc = _assess_cancellation(last_day_cr)
                lines.append(f"  Cancellation Rate: {last_day_cr:.2%}  {_health_badge(ld_cr_status)}  — {ld_cr_desc}")
            if last_day_otp is not None:
                ld_otp_status, ld_otp_desc = _assess_otp(last_day_otp)
                lines.append(f"  On-Time Performance: {last_day_otp:.1%}  {_health_badge(ld_otp_status)}  — {ld_otp_desc}")
        lines.append("")

    # ── Delay summary with per-flight averages ────────────────────
    ds = data.get("delay_summary") or {}
    if ds:
        lines.append("━━━ DELAY SUMMARY ━━━")
        delay_items = [
            ("Departure Delay", "SumDEPARTURE_DELAY"),
            ("Arrival Delay", "SumARRIVAL_DELAY"),
            ("Late Aircraft Delay", "SumLATE_AIRCRAFT_DELAY"),
            ("Airline Delay", "SumAIRLINE_DELAY"),
            ("Weather Delay", "SumWEATHER_DELAY"),
            ("Air System Delay", "SumAIR_SYSTEM_DELAY"),
            ("Security Delay", "SumSECURITY_DELAY"),
        ]
        # Root-cause delay keys (the 5 causes that explain WHY delays happen)
        controllable_keys = [
            "SumLATE_AIRCRAFT_DELAY", "SumAIRLINE_DELAY",
            "SumWEATHER_DELAY", "SumAIR_SYSTEM_DELAY", "SumSECURITY_DELAY",
        ]
        # Total of root-cause delays only (for root-cause % among causes)
        total_root_cause = sum(ds.get(k, 0) or 0 for k in controllable_keys)

        lines.append(f"  {'Delay Type':<25} {'Total (min)':>14} {'Avg/Flight':>12} {'% of Dep':>10} {'% of Causes':>13}")
        lines.append(f"  {'─' * 25} {'─' * 14} {'─' * 12} {'─' * 10} {'─' * 13}")

        total_dep_delay = ds.get("SumDEPARTURE_DELAY", 0) or 0
        total_arr_delay = ds.get("SumARRIVAL_DELAY", 0) or 0

        for label, key in delay_items:
            val = ds.get(key, 0) or 0
            avg_per_flight = (val / total_flights) if total_flights > 0 else 0
            # % of Departure Delay (how much of departure delay does this type represent)
            pct_of_dep = (val / total_dep_delay * 100) if total_dep_delay > 0 else None
            # % of Root Causes (share among the 5 root-cause delay types only)
            is_root_cause = key in controllable_keys
            pct_of_causes = (val / total_root_cause * 100) if (is_root_cause and total_root_cause > 0) else None

            pct_dep_str = f"{pct_of_dep:.1f}%" if pct_of_dep is not None else "   —"
            pct_cause_str = f"{pct_of_causes:.1f}%" if pct_of_causes is not None else "      —"
            lines.append(f"  {label:<25} {_fmt(val):>14} {avg_per_flight:>10.1f}m {pct_dep_str:>10} {pct_cause_str:>13}")

        if total_flights > 0:
            lines.append("")
            lines.append(f"  Average Departure Delay per Flight: {total_dep_delay / total_flights:.1f} minutes")
            lines.append(f"  Average Arrival Delay per Flight: {total_arr_delay / total_flights:.1f} minutes")

        # Root-cause breakdown ranked by share
        ranked_causes = sorted(
            [(key, label, ds.get(key, 0) or 0) for label, key in delay_items if key in controllable_keys],
            key=lambda x: x[2], reverse=True,
        )
        lines.append("")
        lines.append("  ── Outcome Metrics (aggregate delay experienced) ──")
        for out_label, out_key in [("Departure Delay", "SumDEPARTURE_DELAY"), ("Arrival Delay", "SumARRIVAL_DELAY")]:
            val = ds.get(out_key, 0) or 0
            avg = (val / total_flights) if total_flights > 0 else 0
            lines.append(f"  {out_label:<25} {_fmt(val):>12} min  avg {avg:.1f}m/flight  [OUTCOME — not a cause]")

        lines.append("")
        lines.append("  ── Root-Cause Delay Breakdown (% share among 5 delay causes) ──")
        for key, label, minutes in ranked_causes:
            pct = (minutes / total_root_cause * 100) if total_root_cause > 0 else 0
            pct_of_dep = (minutes / total_dep_delay * 100) if total_dep_delay > 0 else 0
            avg = (minutes / total_flights) if total_flights > 0 else 0
            bar = "█" * int(pct / 5)  # rough bar, 1 block per 5%
            lines.append(f"  {label:<25} {pct:>5.1f}% of causes  {pct_of_dep:>5.1f}% of dep  {_fmt(minutes):>12} min  avg {avg:.1f}m/flight  {bar}")

        if ranked_causes:
            top_cause_label = ranked_causes[0][1]
            top_cause_pct = (ranked_causes[0][2] / total_root_cause * 100) if total_root_cause > 0 else 0
            lines.append(f"  🔑 #1 Root Cause: {top_cause_label} ({top_cause_pct:.1f}% of all cause delay minutes)")
            if len(ranked_causes) > 1:
                lines.append(f"  🔑 #2 Root Cause: {ranked_causes[1][1]} ({(ranked_causes[1][2] / total_root_cause * 100) if total_root_cause > 0 else 0:.1f}%)")
        lines.append("")

    # ── Cancellation breakdown with computed shares ───────────────
    cb = data.get("cancellation_breakdown") or []
    if cb:
        lines.append("━━━ CANCELLATIONS BY REASON ━━━")
        total_cancel_sum = sum(r.get("SumCANCELLED", 0) or 0 for r in cb)
        for row in cb:
            reason = row.get("CANCELLATION REASON") or "Unknown"
            count = row.get("SumCANCELLED", 0) or 0
            share = (count / total_cancel_sum * 100) if total_cancel_sum > 0 else 0
            lines.append(f"  {reason}: {_fmt(count)} flights ({share:.1f}%)")
        if total_cancel_sum > 0:
            dominant = max(cb, key=lambda r: r.get("SumCANCELLED", 0) or 0)
            dom_reason = dominant.get("CANCELLATION REASON") or "Unknown"
            dom_count = dominant.get("SumCANCELLED", 0) or 0
            dom_share = dom_count / total_cancel_sum * 100
            lines.append(f"  🔑 Leading cancellation reason: {dom_reason} ({dom_share:.1f}%)")
        lines.append("")

    # ── Flight volume (monthly) with goal comparison ──────────────
    fv = data.get("flight_volume") or []
    vol_monthly: dict | None = None
    if fv:
        vol_monthly = _aggregate_monthly(
            fv, "FLIGHT_DATE",
            {"Actual_Flights": "sum", "Flight_Volume_Goal": "sum"},
        )
        vol_status, vol_desc = _assess_flight_volume(vol_monthly)
        vol_trend = _detect_trend(vol_monthly, "Actual_Flights")
        vol_best_worst = _identify_worst_best_months(vol_monthly, "Actual_Flights", higher_is_better=True)

        lines.append("━━━ FLIGHT VOLUME (monthly) ━━━")
        lines.append(f"  Health: {_health_badge(vol_status)}  {vol_desc}")
        lines.append(f"  Trend: {vol_trend}")
        if vol_best_worst:
            lines.append(f"  {vol_best_worst}")
        lines.append(f"  {'Month':<14} {'Actual':>10} {'Goal':>10} {'Gap':>10} {'Status':>8}")
        lines.append(f"  {'─' * 14} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 8}")
        for month in sorted(vol_monthly):
            vals = vol_monthly[month]
            label = MONTH_LABELS.get(month[-2:], month)
            actual = vals.get("Actual_Flights", 0) or 0
            goal = vals.get("Flight_Volume_Goal", 0) or 0
            gap = actual - goal
            status = "✓" if gap >= 0 else "✗"
            lines.append(f"  {month} ({label}){' ' * (6 - len(label))} {_fmt(actual):>10} {_fmt(goal):>10} {_fmt(gap):>10} {status:>8}")
        lines.append("")

    # ── On-time performance (monthly avg) with assessments ────────
    otp = data.get("otp") or []
    otp_monthly: dict | None = None
    if otp:
        otp_monthly = _aggregate_monthly(
            otp, "FLIGHT_DATE",
            {"On_Time_Performance": "avg", "Target_OTP": "avg"},
        )
        avg_otp_vals = [v.get("On_Time_Performance", 0) or 0 for v in otp_monthly.values()]
        avg_otp = sum(avg_otp_vals) / len(avg_otp_vals) if avg_otp_vals else None
        otp_status, otp_desc = _assess_otp(avg_otp)
        otp_trend = _detect_trend(otp_monthly, "On_Time_Performance")
        otp_best_worst = _identify_worst_best_months(otp_monthly, "On_Time_Performance", higher_is_better=True)

        lines.append("━━━ ON-TIME PERFORMANCE (monthly avg) ━━━")
        lines.append(f"  Health: {_health_badge(otp_status)}  {otp_desc}")
        lines.append(f"  Trend: {otp_trend}")
        if otp_best_worst:
            lines.append(f"  {otp_best_worst}")
        lines.append(f"  {'Month':<14} {'OTP':>8} {'Target':>8} {'Gap':>8} {'Status':>8}")
        lines.append(f"  {'─' * 14} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")
        for month in sorted(otp_monthly):
            vals = otp_monthly[month]
            label = MONTH_LABELS.get(month[-2:], month)
            perf = vals.get("On_Time_Performance", 0) or 0
            target = vals.get("Target_OTP", 0) or 0
            gap = perf - target
            status = "✓" if gap >= 0 else "✗"
            p = f"{perf:.1%}"
            t = f"{target:.1%}"
            g = f"{gap:+.1%}"
            lines.append(f"  {month} ({label}){' ' * (6 - len(label))} {p:>8} {t:>8} {g:>8} {status:>8}")
        lines.append("")

    # ── Cancellation rate (monthly avg) with assessments ──────────
    cr = data.get("cancellation_rate") or []
    cancel_monthly: dict | None = None
    if cr:
        cancel_monthly = _aggregate_monthly(
            cr, "FLIGHT_DATE",
            {"Cancellation_Rate": "avg", "Target_Cancellation": "avg"},
        )
        avg_cancel_vals = [v.get("Cancellation_Rate", 0) or 0 for v in cancel_monthly.values()]
        avg_cancel = sum(avg_cancel_vals) / len(avg_cancel_vals) if avg_cancel_vals else None
        cr_status, cr_desc = _assess_cancellation(avg_cancel)
        cr_trend = _detect_trend(cancel_monthly, "Cancellation_Rate")
        cr_best_worst = _identify_worst_best_months(cancel_monthly, "Cancellation_Rate", higher_is_better=False)

        lines.append("━━━ CANCELLATION RATE (monthly avg) ━━━")
        lines.append(f"  Health: {_health_badge(cr_status)}  {cr_desc}")
        lines.append(f"  Trend: {cr_trend}")
        if cr_best_worst:
            lines.append(f"  {cr_best_worst}")
        lines.append(f"  {'Month':<14} {'Rate':>8} {'Target':>8} {'Gap':>8} {'Status':>8}")
        lines.append(f"  {'─' * 14} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")
        for month in sorted(cancel_monthly):
            vals = cancel_monthly[month]
            label = MONTH_LABELS.get(month[-2:], month)
            rate = vals.get("Cancellation_Rate", 0) or 0
            target = vals.get("Target_Cancellation", 0) or 0
            gap = rate - target
            status = "✓" if gap <= 0 else "✗"
            r = f"{rate:.2%}"
            t = f"{target:.2%}"
            g = f"{gap:+.2%}"
            lines.append(f"  {month} ({label}){' ' * (6 - len(label))} {r:>8} {t:>8} {g:>8} {status:>8}")
        lines.append("")

    # ── SOP RECOMMENDATIONS ───────────────────────────────────────
    sop_lines = _build_sop_recommendations(data, otp_monthly, cancel_monthly, vol_monthly)
    if sop_lines:
        lines.append("=" * 60)
        lines.append("  STANDARD OPERATING PROCEDURE (SOP) RECOMMENDATIONS")
        lines.append("=" * 60)
        lines.extend(sop_lines)

    # ── KPI CONTEXT NOTES (adapted from azure_context.py) ─────────
    lines.append("")
    lines.append("━━━ KPI INTERPRETATION NOTES ━━━")
    lines.append("  • 'Total Flights' Goal: baseline average of successful (non-cancelled) flights.")
    lines.append("    Affected by active slicers (date range, airline); changing slicers changes the baseline.")
    lines.append(f"  • On-Time Performance target: {KPI_GOALS['otp_target']:.0%}. "
                 f"Below {KPI_GOALS['otp_critical']:.0%} is CRITICAL.")
    lines.append(f"  • Cancellation Rate target: {KPI_GOALS['cancellation_target']:.0%}. "
                 f"Above {KPI_GOALS['cancellation_critical']:.0%} is CRITICAL.")
    lines.append("  • Delay values are in MINUTES. Per-flight averages are computed by dividing total delay by total flights.")
    lines.append("  • Monthly OTP and Cancellation Rate are daily-averaged (avg of daily percentages).")
    lines.append("  • Trend direction compares the last third of the period vs the first third.")
    lines.append("")

    # ── Errors
    if data.get("errors"):
        lines.append("━━━ DATA FETCH WARNINGS ━━━")
        for err in data["errors"]:
            lines.append(f"  ⚠ {err}")
        lines.append("")

    return "\n".join(lines)


__all__ = [
    "get_dataset_id",
    "execute_dax",
    "fetch_airline_list",
    "fetch_all_report_data",
    "format_data_for_ai",
]
