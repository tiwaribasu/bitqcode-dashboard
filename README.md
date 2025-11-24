# 📈 IBKR Portfolio Dashboard

A secure, professional dashboard to monitor your Interactive Brokers positions — with **privacy by design**.

- 🔒 Account IDs are masked (e.g., `DU*****67`)
- 🔐 Google Sheet URL stored in **Streamlit Secrets** (never in code)
- 📊 Real-time P&L, allocation charts, and position details
- 🌐 Deployable to **Streamlit Community Cloud** in minutes

---

## 🛠️ Setup

### 1. Publish Your Google Sheet
- In Google Sheets → **File → Share → Publish to web → CSV**
- Copy the URL:  
  `https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/export?format=csv`

### 2. Configure Secrets

#### 🔹 Local Development
Create `.streamlit/secrets.toml`:
```toml
[google_sheet]
csv_url = "https://docs.google.com/spreadsheets/d/.../export?format=csv"