import requests
import streamlit as st
import streamlit.components.v1 as components
from .config import TENANT_ID, CLIENT_ID, CLIENT_SECRET, WORKSPACE_ID, REPORT_ID


def get_aad_token() -> str:
    """Acquire an Azure AD access token for the Power BI REST API.

    This token is used server-side for embed-token generation **and**
    for DAX ``executeQueries`` calls.
    """
    auth_url = (
        f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
    )
    auth_data = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "scope": "https://analysis.windows.net/powerbi/api/.default",
    }
    auth_res = requests.post(auth_url, data=auth_data, timeout=30)
    if auth_res.status_code != 200:
        raise ConnectionError(
            f"Azure AD auth failed ({auth_res.status_code}): {auth_res.text}"
        )
    return auth_res.json()["access_token"]


def get_embed_params():
    # 1. Get Azure AD Access Token
    try:
        aad_token = get_aad_token()
    except ConnectionError as exc:
        st.error(str(exc))
        st.stop()

    # 2. Get Report Details
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {aad_token}'}
    report_url = f"https://api.powerbi.com/v1.0/myorg/groups/{WORKSPACE_ID}/reports/{REPORT_ID}"
    
    report_res = requests.get(report_url, headers=headers, timeout=30)
    
    if report_res.status_code != 200:
        st.error(f"Power BI API Failed ({report_res.status_code})")
        st.info("Check if your Service Principal (App) is a Member of the Power BI Workspace.")
        st.write("Full Error from Microsoft:", report_res.text)
        return None, None, None

    embed_data = report_res.json()

    # 3. Generate an embed token
    generate_url = f"https://api.powerbi.com/v1.0/myorg/groups/{WORKSPACE_ID}/reports/{REPORT_ID}/GenerateToken"
    generate_body = {"accessLevel": "View"}
    generate_res = requests.post(generate_url, headers=headers, json=generate_body, timeout=30)

    if generate_res.status_code != 200:
        st.error(f"Generate Embed Token Failed ({generate_res.status_code})")
        try:
            st.json(generate_res.json())
        except Exception:
            st.write(generate_res.text)
        return None, None, None

    embed_token = generate_res.json().get("token")
    return embed_data.get("embedUrl"), embed_token, embed_data.get("datasetId")


def render_powerbi_report(embed_token: str, embed_url: str, airline_filter: str = None, date_from: str = None, date_to: str = None):
    """
    Render Power BI report with optional airline filter applied on load.
    """
    df_val = date_from or ""
    dt_val = date_to or ""
    af_val = airline_filter or ""
    filter_js = f"""
        report.on('loaded', async function() {{
            try {{
                let airlineSlicerFound = false;
                let dateSlicerFound = false;

                try {{
                    const pages = await report.getPages();
                    for (const p of pages) {{
                        const visuals = await p.getVisuals();
                        for (const v of visuals) {{
                            if (v.type === 'slicer') {{
                                try {{
                                    const state = await v.getSlicerState();
                                    const targets = state.targets || [];

                                    for (const target of targets) {{
                                        if (target.column === 'AIRLINE_NAME') {{
                                            airlineSlicerFound = true;
                                            try {{
                                                if ("{af_val}" && "{af_val}" !== "") {{
                                                    const airlineFilter = {{
                                                        $schema: "http://powerbi.com/product/schema#basic",
                                                        target: {{ table: target.table, column: target.column }},
                                                        operator: "In",
                                                        values: ["{af_val}"],
                                                        filterType: models.FilterType.BasicFilter
                                                    }};
                                                    await v.setSlicerState({{ filters: [airlineFilter] }});
                                                }} else {{
                                                    await v.setSlicerState({{ filters: [] }});
                                                }}
                                            }} catch (err) {{
                                                airlineSlicerFound = false;
                                            }}
                                        }}

                                        if (target.column === 'FLIGHT_DATE') {{
                                            dateSlicerFound = true;
                                            try {{
                                                if ("{df_val}" || "{dt_val}") {{
                                                    let filterApplied = false;
                                                    const formats = [
                                                        () => {{
                                                            const filter = {{
                                                                $schema: "http://powerbi.com/product/schema#advanced",
                                                                target: {{ table: target.table, column: target.column }},
                                                                logicalOperator: "And",
                                                                conditions: [],
                                                                filterType: models.FilterType.AdvancedFilter
                                                            }};
                                                            if ("{df_val}") filter.conditions.push({{ operator: "GreaterThanOrEqual", value: "{df_val}T00:00:00" }});
                                                            if ("{dt_val}") filter.conditions.push({{ operator: "LessThanOrEqual", value: "{dt_val}T23:59:59" }});
                                                            return filter;
                                                        }},
                                                        () => {{
                                                            const filter = {{
                                                                $schema: "http://powerbi.com/product/schema#advanced",
                                                                target: {{ table: target.table, column: target.column }},
                                                                logicalOperator: "And",
                                                                conditions: [],
                                                                filterType: models.FilterType.AdvancedFilter
                                                            }};
                                                            if ("{df_val}") filter.conditions.push({{ operator: "GreaterThanOrEqual", value: new Date("{df_val}T00:00:00Z") }});
                                                            if ("{dt_val}") filter.conditions.push({{ operator: "LessThanOrEqual", value: new Date("{dt_val}T23:59:59Z") }});
                                                            return filter;
                                                        }},
                                                        () => {{
                                                            const filter = {{
                                                                $schema: "http://powerbi.com/product/schema#advanced",
                                                                target: {{ table: target.table, column: target.column }},
                                                                logicalOperator: "And",
                                                                conditions: [],
                                                                filterType: models.FilterType.AdvancedFilter
                                                            }};
                                                            if ("{df_val}") filter.conditions.push({{ operator: "GreaterThanOrEqual", value: new Date("{df_val}T00:00:00Z").getTime() }});
                                                            if ("{dt_val}") filter.conditions.push({{ operator: "LessThanOrEqual", value: new Date("{dt_val}T23:59:59Z").getTime() }});
                                                            return filter;
                                                        }}
                                                    ];

                                                    for (let i = 0; i < formats.length && !filterApplied; i++) {{
                                                        try {{
                                                            const dateFilter = formats[i]();
                                                            await v.setSlicerState({{ filters: [dateFilter] }});
                                                            filterApplied = true;
                                                        }} catch (err) {{
                                                        }}
                                                    }}

                                                    if (!filterApplied) {{
                                                        dateSlicerFound = false;
                                                    }}
                                                }} else {{
                                                    await v.setSlicerState({{ filters: [] }});
                                                }}
                                            }} catch (err) {{
                                                dateSlicerFound = false;
                                            }}
                                        }}
                                    }}
                                }} catch (err) {{
                                }}
                            }}
                        }}
                    }}
                }} catch (err) {{
                }}

                const reportFilters = [];

                if (!airlineSlicerFound && "{af_val}" && "{af_val}" !== "") {{
                    reportFilters.push({{
                        $schema: "http://powerbi.com/product/schema#basic",
                        target: {{ table: "gold_aviation_report", column: "AIRLINE_NAME" }},
                        operator: "In",
                        values: ["{af_val}"],
                        filterType: models.FilterType.BasicFilter
                    }});
                }}

                if (!dateSlicerFound && ("{df_val}" || "{dt_val}")) {{
                    const dateFilter = {{
                        $schema: "http://powerbi.com/product/schema#advanced",
                        target: {{ table: "gold_aviation_report", column: "FLIGHT_DATE" }},
                        logicalOperator: "And",
                        conditions: [],
                        filterType: models.FilterType.AdvancedFilter
                    }};
                    if ("{df_val}") {{
                        dateFilter.conditions.push({{ operator: "GreaterThanOrEqual", value: "{df_val}T00:00:00" }});
                    }}
                    if ("{dt_val}") {{
                        dateFilter.conditions.push({{ operator: "LessThanOrEqual", value: "{dt_val}T23:59:59" }});
                    }}
                    if (dateFilter.conditions.length > 0) {{
                        reportFilters.push(dateFilter);
                    }}
                }}

                if (reportFilters.length > 0) {{
                    await report.setFilters(reportFilters);
                }} else if (!"{af_val}" && !"{df_val}" && !"{dt_val}") {{
                    await report.removeFilters();
                }}
            }} catch (err) {{
            }}
        }});
    """
    
    pbi_html = f"""
    <div id="reportContainer" style="height: 600px; width: 100%;"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/powerbi-client/2.19.1/powerbi.min.js"></script>
    <script>
        var models = window['powerbi-client'].models;
        var config = {{
            type: 'report',
            tokenType: models.TokenType.Embed,
            accessToken: '{embed_token}',
            embedUrl: '{embed_url}',
            id: '{REPORT_ID}',
            settings: {{
                filterPaneEnabled: false,
                navContentPaneEnabled: true
            }}
        }};

        var reportContainer = document.getElementById('reportContainer');
        var report = powerbi.embed(reportContainer, config);
        
        {filter_js}
    </script>
    """
    components.html(pbi_html, height=650)

__all__ = ["get_aad_token", "get_embed_params", "render_powerbi_report"]
