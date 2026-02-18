# Flight-Report

Flight-Report is a Streamlit app that directly queries a Power BI dataset using the Power BI REST API's DAX `executeQueries` endpoint. The app uses a service principal (App Registration) to authenticate, generates an embed token to display the report, and runs DAX queries to fetch the same data shown in the report visuals.

**Important note:** This repository now uses a DAX-based approach (server-side `executeQueries`) instead of a vision pipeline. The DAX approach requires your Power BI workspace to be on Premium, Premium Per User (PPU), or Embedded capacity; otherwise the `executeQueries` API will not work.

**Prerequisites:**

- An **Azure AD** App Registration (service principal) with application credentials.
- Power BI workspace that is **Premium**, **PPU**, or **Embedded**.
- Service principal must have the appropriate Power BI API application permission (e.g., `Dataset.Read.All`) and admin consent granted.
- The service principal (or a security group containing it) must be added to the target Power BI Workspace as a Member/Contributor.
- Python 3.10+ and virtual environment for running the Streamlit app.

**High-level setup:**

1. Create an Azure App Registration and record Tenant ID, Client (Application) ID, and Client Secret.
2. Grant the app the Power BI application permissions (e.g., `Dataset.Read.All`) and grant admin consent.
3. Add the service principal (or its security group) to the Power BI Workspace.
4. Configure environment variables in a `.env` file.
5. Install dependencies and run the Streamlit app (`app.py`).

**1) Get Azure Credentials (Tenant, Client, Secret)**

1. In the Azure Portal, open **App registrations** → **New registration**.
   - Name: something like `Streamlit_PowerBI_Authority`.
   - Supported account types: "Accounts in this organizational directory only".
   - Redirect URI not required for server-side client credentials flow (you can leave it blank or set to `http://localhost:8501`).
2. After creating the app, copy the **Application (client) ID** and **Directory (tenant) ID** from the Overview page.
3. Under **Certificates & secrets**, create a **New client secret** and copy its value immediately.

**2) API permissions & admin consent**

1. In the App Registration, open **API permissions** → **Add a permission** → **Power BI Service**.
2. Choose **Application permissions** and add `Dataset.Read.All` (or `Dataset.ReadWrite.All` if needed).
3. Ask a tenant admin to **Grant admin consent** for the permission.

**3) Add the service principal to your Power BI Workspace**

1. (Optional but recommended) Create an Azure AD Security Group (e.g., `AI_PowerBI_Access`) and add the App Registration as a member.
2. In the Power BI Service, open your Workspace → **Manage access** → **Add people or groups** and add the service principal or the security group as a Member or Contributor.
3. In the Power BI Admin Portal, ensure Tenant settings allow service principals to use the Power BI APIs (Tenant admin action):
   - Tenant settings → Developer settings → Enable "Allow service principals to use Power BI APIs" and apply to the security group.

**Capacity requirement**
The `executeQueries` REST API requires the workspace to be on Power BI Premium, PPU, or Embedded capacity. If you attempt to run DAX queries against a non-Premium workspace you'll receive helpful runtime errors. See `app_lib/powerbi_data.py` for error handling and messages.

**Environment variables (.env)**
The app expects the following keys (defined in `app_lib/config.py`):

- `POWERBI_TENANT_ID` — Azure AD tenant ID.
- `POWERBI_CLIENT_ID` — App (client) ID.
- `POWERBI_CLIENT_SECRET` — Client secret value.
- `POWERBI_WORKSPACE_ID` — Power BI Workspace (group) ID.
- `POWERBI_REPORT_ID` — Power BI Report ID.

Example `.env`:

```
POWERBI_TENANT_ID=00000000-0000-0000-0000-000000000000
POWERBI_CLIENT_ID=11111111-1111-1111-1111-111111111111
POWERBI_CLIENT_SECRET=your-secret-value
POWERBI_WORKSPACE_ID=<WORKSPACE_ID_FROM_PBI_URL>
POWERBI_REPORT_ID=<REPORT_ID_FROM_PBI_URL>
```

Files of interest:

- `app.py` — Main Streamlit app and chat UI ([app.py](app.py)).
- `app_lib/config.py` — Loads environment variables ([app_lib/config.py](app_lib/config.py)).
- `app_lib/powerbi.py` — Acquire AAD token, generate embed token, and render the report ([app_lib/powerbi.py](app_lib/powerbi.py)).
- `app_lib/powerbi_data.py` — Executes DAX queries via `executeQueries` to fetch report data ([app_lib/powerbi_data.py](app_lib/powerbi_data.py)).

**Install and Run (local, Windows example)**

1. Create and activate a virtual environment:

```powershell
python -m venv venv
venv\\Scripts\\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the Streamlit app (entrypoint is `app.py`):

```powershell
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

**Troubleshooting & common errors**

- DAX timeouts or errors mentioning "premium" or "capacity": your workspace likely isn't Premium/PPU/Embedded — `executeQueries` requires Premium capacity.
- `401` / AAD auth failures: confirm `POWERBI_TENANT_ID`, `POWERBI_CLIENT_ID`, and `POWERBI_CLIENT_SECRET` are correct and the app has **application** permissions with admin consent.
- Permission denied when calling the Power BI Report or GenerateToken endpoints: ensure the service principal (or security group) is added to the workspace with at least Member access.
- If `get_embed_params()` fails, check the printed API error in the Streamlit sidebar — it often contains the provider's JSON error message.

**Security & production notes**

- Do not commit `.env` or secrets to source control. Use a secrets store (Azure Key Vault, environment-managed secrets) in production.
- For production deployment, consider using managed identities, rotating keys, and least-privilege permissions.
