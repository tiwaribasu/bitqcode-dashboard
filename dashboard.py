import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, date
import pytz
from news_sentiment_module import NewsSentimentAnalyzer

# ===================================================================
# 🛠️ CONFIGURATION
# ===================================================================
REFRESH_INTERVAL_SEC = 10
CACHE_TTL = 10

# ===================================================================
# 🔐 Load Google Sheet URL from Streamlit Secrets
# ===================================================================
try:
    GOOGLE_SHEET_CSV_URL = st.secrets["google_sheet"]["csv_url"]
    NEWS_SHEET_URL = st.secrets.get("news_sheet", {}).get("url", "")

except KeyError:
    st.error("🔐 Missing Google Sheet URL in Streamlit Secrets.")
    st.stop()

st.set_page_config(
    page_title="BITQCODE Dashboard",
    page_icon="💼",
    layout="wide"
)

# st._config.set_option('client.showErrorDetails', True)


# ===================================================================
# 🧮 Helper Functions
# ===================================================================
def mask_account(acc: str) -> str:
    if not isinstance(acc, str) or len(acc) < 5:
        return "N/A"
    return acc[:2] + "*****" + acc[-2:]

def format_currency(val, currency_symbol="$"):
    if pd.isna(val) or val == 0:
        return f"{currency_symbol}0.00"
    return f"{currency_symbol}{val:,.2f}"

def format_inr(val):
    """Format Indian Rupees"""
    if pd.isna(val) or val == 0:
        return "₹0.00"
    return f"₹{val:,.2f}"

def format_percent(val):
    if pd.isna(val):
        return "—"
    return f"{val:+.2f}%"

def get_time_with_timezone(region):
    """Get current time with appropriate timezone"""
    if region == "INDIA":
        tz = pytz.timezone('Asia/Kolkata')
        now = datetime.now(tz)
        return now.strftime('%Y-%m-%d %H:%M:%S IST')
    else:
        tz = pytz.timezone('US/Eastern')
        now = datetime.now(tz)
        return now.strftime('%Y-%m-%d %H:%M:%S ET')

def get_currency_formatter(region):
    """Return appropriate currency formatter based on region"""
    return format_inr if region == "INDIA" else lambda x: format_currency(x, "$")

def get_currency_symbol(region):
    """Return currency symbol based on region"""
    return "₹" if region == "INDIA" else "$"

def create_metric_card(title, value, value_color="#000000"):
    """Create a metric card with consistent styling"""
    return f"""
    <div style="text-align: center;">
        <div style="font-size: 0.85rem; font-weight: 600; color: {value_color}; margin-bottom: 0.2rem;">{title}</div>
        <div style="font-size: 1.5rem; font-weight: 700; color: {value_color};">{value}</div>
    </div>
    """

def get_pnl_color(value, currency_symbol):
    """Return color for P&L values"""
    if pd.isna(value) or value == 0:
        return "gray"
    str_value = str(value)
    if f"{currency_symbol}-" in str_value or "-" in str_value or "−" in str_value:
        return "red"
    return "green"

def create_html_table(df, columns, currency_symbol):
    """Create HTML table from dataframe with consistent styling"""
    if df.empty:
        return ""
    
    # Create header
    html = """
    <div style="overflow-x: auto;">
    <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
        <thead>
            <tr style="background-color: #f2f2f2;">
    """
    
    for col in columns:
        align = "right" if col in ['Quantity', 'Avg Price', 'Last Price', 'Unrealized P&L', 'Open Exposure', 
                                  'Buy Qty', 'Buy Price', 'Sell Qty', 'Sell Price', 'Realized P&L'] else "left"
        html += f'<th style="padding: 10px; text-align: {align}; border-bottom: 1px solid #ddd;">{col}</th>'
    
    html += """
            </tr>
        </thead>
        <tbody>
    """
    
    # Add rows
    for _, row in df.iterrows():
        html += '<tr>'
        for col in columns:
            align = "right" if col in ['Quantity', 'Avg Price', 'Last Price', 'Unrealized P&L', 'Open Exposure',
                                      'Buy Qty', 'Buy Price', 'Sell Qty', 'Sell Price', 'Realized P&L'] else "left"
            
            cell_value = row[col]
            cell_style = f"padding: 8px; border-bottom: 1px solid #ddd; text-align: {align};"
            
            # Apply color coding for P&L columns
            if col in ['Unrealized P&L', 'Realized P&L']:
                pnl_color = get_pnl_color(cell_value, currency_symbol)
                cell_style += f" font-weight: bold; color: {pnl_color};"
            
            html += f'<td style="{cell_style}">{cell_value}</td>'
        
        html += '</tr>'
    
    html += """
            </tbody>
        </table>
    </div>
    """
    
    return html

# ===================================================================
# 📥 Data Loading & Processing
# ===================================================================
@st.cache_data(ttl=REFRESH_INTERVAL_SEC, show_spinner=False)
def load_sheet_data(sheet_gid="0"):
    """Load specific sheet from Google Sheets using gid parameter"""
    try:
        if "export?format=csv" in GOOGLE_SHEET_CSV_URL:
            if "gid=" in GOOGLE_SHEET_CSV_URL:
                url = GOOGLE_SHEET_CSV_URL.split("&gid=")[0] + f"&gid={sheet_gid}"
            else:
                url = GOOGLE_SHEET_CSV_URL + f"&gid={sheet_gid}"
        else:
            url = GOOGLE_SHEET_CSV_URL + f"?gid={sheet_gid}&format=csv"
        
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"❌ Failed to load sheet {sheet_gid}: {str(e)[:150]}...")
        return pd.DataFrame()

@st.cache_data(ttl=REFRESH_INTERVAL_SEC)
def process_live_pnl_data(df_raw):
    """Process Live PnL data - filter for latest date only"""
    if df_raw.empty:
        return pd.DataFrame()
    
    df = df_raw.copy()
    df.columns = df.columns.str.strip()
    
    required_cols = ['DateTime', 'Total PnL']
    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame()
    
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    df['Total PnL'] = pd.to_numeric(df['Total PnL'], errors='coerce')
    df = df.dropna(subset=['DateTime', 'Total PnL'])
    
    if df.empty:
        return df
    
    df['Date'] = df['DateTime'].dt.date
    latest_date = df['Date'].max()
    df_today = df[df['Date'] == latest_date].copy()
    df_today = df_today.sort_values('DateTime')
    
    return df_today

@st.cache_data(ttl=REFRESH_INTERVAL_SEC)
def process_india_data(df_raw):
    """Process INDIA data with the new format"""
    if df_raw.empty:
        return {
            'open_positions': pd.DataFrame(),
            'closed_positions': pd.DataFrame(),
            'summary': {}
        }
    
    df = df_raw.copy()
    df.columns = df.columns.str.strip()
    
    expected_cols = [
        's_no', 'tradingsymbol', 'buy_value', 'buy_price', 
        'buy_quantity', 'sell_quantity', 'sell_price', 
        'sell_value', 'last_price', 'pnl'
    ]
    
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        return {
            'open_positions': pd.DataFrame(),
            'closed_positions': pd.DataFrame(),
            'summary': {}
        }
    
    numeric_cols = ['buy_value', 'buy_price', 'buy_quantity', 'sell_quantity', 
                   'sell_price', 'sell_value', 'last_price', 'pnl']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['tradingsymbol'])
    df = df[df['tradingsymbol'].astype(str).str.strip() != '']
    
    # Separate open and closed positions
    closed_mask = (df['buy_quantity'] > 0) & (df['sell_quantity'] > 0) & (df['buy_quantity'] == df['sell_quantity'])
    closed_df = df[closed_mask].copy()
    open_mask = ~closed_mask
    open_df = df[open_mask].copy()
    
    # Calculate additional metrics for open positions
    if not open_df.empty:
        open_df['net_quantity'] = open_df['buy_quantity'] - open_df['sell_quantity']
        open_df['avg_price'] = np.where(
            open_df['net_quantity'] != 0,
            (open_df['buy_value'] - open_df['sell_value']) / open_df['net_quantity'],
            0
        )
        open_df['unrealized_pnl'] = (open_df['last_price'] - open_df['avg_price']) * open_df['net_quantity']
        open_df['open_exposure'] = open_df['net_quantity'] * open_df['last_price']
        open_df['position_type'] = open_df['net_quantity'].apply(lambda x: 'Long' if x > 0 else 'Short' if x < 0 else 'Flat')
        open_df = open_df.sort_values('unrealized_pnl', ascending=False)
    
    # Calculate summary metrics
    total_traded_volume = df['buy_value'].sum() + df['sell_value'].sum()
    total_closed_pnl = closed_df['pnl'].sum() if not closed_df.empty else 0
    total_unrealized_pnl = open_df['unrealized_pnl'].sum() if not open_df.empty else 0
    total_open_exposure = open_df['open_exposure'].abs().sum() if not open_df.empty else 0
    
    return {
        'open_positions': open_df,
        'closed_positions': closed_df,
        'summary': {
            'total_traded_volume': total_traded_volume,
            'total_closed_pnl': total_closed_pnl,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_open_exposure': total_open_exposure,
            'open_positions_count': len(open_df),
            'closed_positions_count': len(closed_df),
            'total_pnl': total_closed_pnl + total_unrealized_pnl
        }
    }


# @st.cache_data(ttl=REFRESH_INTERVAL_SEC)
# def process_india_data(df_raw):
#     if df_raw.empty:
#         return {'open_positions': pd.DataFrame(), 'closed_positions': pd.DataFrame(), 'summary': {}}

#     df = df_raw.copy()
#     df.columns = df.columns.str.strip()

#     numeric_cols = ['buy_value','buy_price','buy_quantity','sell_quantity','sell_price','sell_value','last_price','pnl']
#     for col in numeric_cols:
#         df[col] = pd.to_numeric(df[col], errors='coerce')

#     df = df.dropna(subset=['tradingsymbol'])
#     df = df[df['tradingsymbol'].astype(str).str.strip() != '']

#     closed_mask = (df['buy_quantity'] > 0) & (df['sell_quantity'] > 0) & (df['buy_quantity'] == df['sell_quantity'])
#     closed_df = df[closed_mask].copy()
#     open_df = df[~closed_mask].copy()

#     # ---------------- FORCE MATHEMATICAL REALIZED PNL ----------------
#     if not closed_df.empty:
#         closed_df['pnl'] = (closed_df['sell_price'] - closed_df['buy_price']) * closed_df['buy_quantity']

#     # ---------------- OPEN POSITIONS ----------------
#     if not open_df.empty:
#         open_df['net_quantity'] = open_df['buy_quantity'] - open_df['sell_quantity']
#         open_df['avg_price'] = np.where(open_df['net_quantity'] != 0,
#                                          (open_df['buy_value'] - open_df['sell_value']) / open_df['net_quantity'], 0)
#         open_df['unrealized_pnl'] = (open_df['last_price'] - open_df['avg_price']) * open_df['net_quantity']
#         open_df['open_exposure'] = open_df['net_quantity'] * open_df['last_price']
#         open_df['position_type'] = open_df['net_quantity'].apply(lambda x: 'Long' if x > 0 else 'Short' if x < 0 else 'Flat')
#         open_df = open_df.sort_values('unrealized_pnl', ascending=False)

#     # ---------------- SUMMARY ----------------
#     total_traded_volume = df['buy_value'].sum() + df['sell_value'].sum()
#     total_closed_pnl = closed_df['pnl'].sum() if not closed_df.empty else 0
#     total_unrealized_pnl = open_df['unrealized_pnl'].sum() if not open_df.empty else 0
#     total_open_exposure = open_df['open_exposure'].abs().sum() if not open_df.empty else 0
#     total_pnl = total_closed_pnl + total_unrealized_pnl

#     return {
#         'open_positions': open_df,
#         'closed_positions': closed_df,
#         'summary': {
#             'total_traded_volume': total_traded_volume,
#             'total_closed_pnl': total_closed_pnl,
#             'total_unrealized_pnl': total_unrealized_pnl,
#             'total_open_exposure': total_open_exposure,
#             'open_positions_count': len(open_df),
#             'closed_positions_count': len(closed_df),
#             'total_pnl': total_pnl
#         }
#     }



@st.cache_data(ttl=REFRESH_INTERVAL_SEC)
def process_daily_pnl_data(df_raw, region="INDIA"):
    """Process Daily PnL data for INDIA and GLOBAL"""
    if df_raw.empty:
        return pd.DataFrame()
    
    df = df_raw.copy()
    df.columns = df.columns.str.strip()
    
    required_cols = ['Date', 'Gross P&L', 'Charges', 'Net P&L']
    
    # Try alternative column names
    alternative_cols = {
        'Gross P&L': ['Gross PnL', 'GrossPnL', 'Gross'],
        'Charges': ['Fees', 'Commission', 'Brokerage'],
        'Net P&L': ['Net PnL', 'NetPnL', 'Net']
    }
    
    for required, alternatives in alternative_cols.items():
        if required not in df.columns:
            for alt in alternatives:
                if alt in df.columns:
                    df[required] = df[alt]
                    break
    
    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame()
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    
    for col in ['Gross P&L', 'Charges', 'Net P&L']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.sort_values('Date', ascending=False)
    df['Cumulative Gross P&L'] = df['Gross P&L'].cumsum()
    df['Cumulative Net P&L'] = df['Net P&L'].cumsum()
    df['Date_Display'] = df['Date'].dt.strftime('%Y-%m-%d')
    df['Region'] = region
    
    return df

# ===================================================================
# 📊 Dashboard Creation Functions
# ===================================================================
def create_live_pnl_chart(live_pnl_df, currency_symbol):
    """Create live P&L chart with color transitions"""
    if live_pnl_df.empty:
        return None
    
    live_pnl_df_sorted = live_pnl_df.sort_values('DateTime')
    highest_value = live_pnl_df_sorted['Total PnL'].max()
    lowest_value = live_pnl_df_sorted['Total PnL'].min()
    highest_row = live_pnl_df_sorted[live_pnl_df_sorted['Total PnL'] == highest_value].iloc[0]
    lowest_row = live_pnl_df_sorted[live_pnl_df_sorted['Total PnL'] == lowest_value].iloc[0]
    
    fig = go.Figure()
    segments = []
    current_segment = {'x': [], 'y': [], 'color': None}
    
    for i in range(len(live_pnl_df_sorted)):
        current_val = live_pnl_df_sorted['Total PnL'].iloc[i]
        current_time = live_pnl_df_sorted['DateTime'].iloc[i]
        current_color = '#10B981' if current_val >= 0 else '#EF4444'
        
        if not current_segment['x']:
            current_segment['x'].append(current_time)
            current_segment['y'].append(current_val)
            current_segment['color'] = current_color
        elif current_segment['color'] == current_color:
            current_segment['x'].append(current_time)
            current_segment['y'].append(current_val)
        else:
            prev_val = live_pnl_df_sorted['Total PnL'].iloc[i-1]
            prev_time = live_pnl_df_sorted['DateTime'].iloc[i-1]
            m = (current_val - prev_val) / ((current_time - prev_time).total_seconds())
            zero_time_seconds = -prev_val / m if m != 0 else 0
            zero_time = prev_time + pd.Timedelta(seconds=zero_time_seconds)
            
            current_segment['x'].append(zero_time)
            current_segment['y'].append(0)
            segments.append(current_segment.copy())
            current_segment = {
                'x': [zero_time, current_time],
                'y': [0, current_val],
                'color': current_color
            }
    
    if current_segment['x']:
        segments.append(current_segment)
    
    for segment in segments:
        fig.add_trace(go.Scatter(
            x=segment['x'],
            y=segment['y'],
            mode='lines',
            line=dict(shape='spline', smoothing=1.0, width=3, color=segment['color']),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add invisible trace for hover
    fig.add_trace(go.Scatter(
        x=live_pnl_df_sorted['DateTime'],
        y=live_pnl_df_sorted['Total PnL'],
        mode='lines',
        line=dict(width=0),
        hovertemplate=f'<b>%{{x|%H:%M:%S}}</b><br>{currency_symbol}%{{y:,.2f}}<extra></extra>',
        showlegend=False,
        name='Live P&L'
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="#94A3B8", line_width=1, opacity=0.3)
    
    # Add area fill
    x_full = live_pnl_df_sorted['DateTime'].tolist()
    y_full = live_pnl_df_sorted['Total PnL'].tolist()
    
    fig.add_trace(go.Scatter(
        x=x_full,
        y=y_full,
        mode='none',
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.1)',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=x_full,
        y=[min(y, 0) for y in y_full],
        mode='none',
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.1)',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add extreme points
    fig.add_trace(go.Scatter(
        x=[highest_row['DateTime']],
        y=[highest_value],
        mode='markers+text',
        marker=dict(size=12, color='#10B981', symbol='triangle-up', line=dict(width=2, color='white')),
        text=[f"  High: {currency_symbol}{highest_value:,.0f}"],
        textposition="top center",
        textfont=dict(size=11, color='#10B981', family='Arial'),
        hovertemplate=f'<b>Highest: {currency_symbol}{highest_value:,.2f}</b><br>Time: %{{x|%H:%M:%S}}<extra></extra>',
        showlegend=False,
        name='Highest'
    ))
    
    fig.add_trace(go.Scatter(
        x=[lowest_row['DateTime']],
        y=[lowest_value],
        mode='markers+text',
        marker=dict(size=12, color='#EF4444', symbol='triangle-down', line=dict(width=2, color='white')),
        text=[f"  Low: {currency_symbol}{lowest_value:,.0f}"],
        textposition="bottom center",
        textfont=dict(size=11, color='#EF4444', family='Arial'),
        hovertemplate=f'<b>Lowest: {currency_symbol}{lowest_value:,.2f}</b><br>Time: %{{x|%H:%M:%S}}<extra></extra>',
        showlegend=False,
        name='Lowest'
    ))
    
    # Update layout
    fig.update_layout(
        height=380,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter, system-ui, sans-serif", size=12),
        hovermode='x unified',
        margin=dict(l=0, r=0, t=20, b=40),
        xaxis=dict(
            showgrid=False,
            tickformat='%H:%M',
            tickfont=dict(size=10, color='#64748B'),
            linecolor='#E2E8F0',
            showline=True
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#F1F5F9',
            gridwidth=1,
            tickprefix=currency_symbol,
            tickformat=',.0f',
            tickfont=dict(size=10, color='#64748B'),
            linecolor='#E2E8F0',
            showline=True
        ),
        showlegend=False
    )
    
    return fig

def create_daily_pnl_chart(daily_pnl_df, currency_symbol):
    """Create daily P&L chart with capital line"""
    if daily_pnl_df.empty or 'Capital' not in daily_pnl_df.columns:
        return None
    
    daily_pnl_df_sorted = daily_pnl_df.sort_values('Date', ascending=True).copy()
    daily_pnl_df_sorted['Date_Str'] = daily_pnl_df_sorted['Date'].dt.strftime('%Y-%m-%d')
    
    highest_capital = daily_pnl_df_sorted['Capital'].max()
    lowest_capital = daily_pnl_df_sorted['Capital'].min()
    highest_capital_idx = daily_pnl_df_sorted['Capital'].idxmax()
    lowest_capital_idx = daily_pnl_df_sorted['Capital'].idxmin()
    latest_capital = daily_pnl_df_sorted['Capital'].iloc[-1]
    latest_date_str = daily_pnl_df_sorted['Date_Str'].iloc[-1]
    
    fig = go.Figure()
    
    # Add Net P&L as colored bars
    colors = ['#10B981' if val >= 0 else '#EF4444' for val in daily_pnl_df_sorted['Net P&L']]
    fig.add_trace(go.Bar(
        x=daily_pnl_df_sorted['Date_Str'],
        y=daily_pnl_df_sorted['Net P&L'],
        name='Daily Net P&L',
        marker_color=colors,
        opacity=0.8,
        hovertemplate=f'<b>%{{x}}</b><br>Net P&L: {currency_symbol}%{{y:,.2f}}<extra></extra>',
        yaxis='y'
    ))
    
    # Add Capital line
    fig.add_trace(go.Scatter(
        x=daily_pnl_df_sorted['Date_Str'],
        y=daily_pnl_df_sorted['Capital'],
        name='Capital',
        mode='lines+markers',
        line=dict(color='#FFA500', width=3),
        marker=dict(size=6, color='#FFA500'),
        hovertemplate=f'<b>%{{x}}</b><br>Capital: {currency_symbol}%{{y:,.2f}}<extra></extra>',
        yaxis='y2'
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="#94A3B8", line_width=1, opacity=0.3, yref="y")
    
    # Add extreme markers
    if not pd.isna(highest_capital_idx):
        highest_capital_date_str = daily_pnl_df_sorted.loc[highest_capital_idx, 'Date_Str']
        fig.add_trace(go.Scatter(
            x=[highest_capital_date_str],
            y=[highest_capital],
            mode='markers+text',
            marker=dict(size=14, color='#FFA500', symbol='triangle-up', line=dict(width=2, color='white')),
            text=[f"  High: {currency_symbol}{highest_capital:,.0f}"],
            textposition="top center",
            textfont=dict(size=11, color='#FFA500', family='Arial'),
            hovertemplate=f'<b>Highest Capital: {currency_symbol}{highest_capital:,.2f}</b><br>Date: %{{x}}<extra></extra>',
            showlegend=False,
            name='Highest Capital',
            yaxis='y2'
        ))
    
    if not pd.isna(lowest_capital_idx):
        lowest_capital_date_str = daily_pnl_df_sorted.loc[lowest_capital_idx, 'Date_Str']
        fig.add_trace(go.Scatter(
            x=[lowest_capital_date_str],
            y=[lowest_capital],
            mode='markers+text',
            marker=dict(size=14, color='#FFA500', symbol='triangle-down', line=dict(width=2, color='white')),
            text=[f"  Low: {currency_symbol}{lowest_capital:,.0f}"],
            textposition="bottom center",
            textfont=dict(size=11, color='#FFA500', family='Arial'),
            hovertemplate=f'<b>Lowest Capital: {currency_symbol}{lowest_capital:,.2f}</b><br>Date: %{{x}}<extra></extra>',
            showlegend=False,
            name='Lowest Capital',
            yaxis='y2'
        ))
    
    # Add current capital marker
    fig.add_trace(go.Scatter(
        x=[latest_date_str],
        y=[latest_capital],
        mode='markers+text',
        marker=dict(size=10, color='#FFA500', symbol='diamond', line=dict(width=2, color='white')),
        text=[f"  Current: {currency_symbol}{latest_capital:,.0f}"],
        textposition="top right",
        textfont=dict(size=11, color='#FFA500', family='Arial'),
        hovertemplate=f'<b>Current Capital: {currency_symbol}{latest_capital:,.2f}</b><br>Date: %{{x}}<extra></extra>',
        showlegend=False,
        name='Current Capital',
        yaxis='y2'
    ))
    
    # Update layout
    fig.update_layout(
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter, system-ui, sans-serif", size=12),
        hovermode='x unified',
        margin=dict(l=60, r=60, t=40, b=60),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        bargap=0.2,
        xaxis=dict(
            title="Date",
            type='category',
            categoryorder='category ascending',
            showgrid=False,
            tickfont=dict(size=10, color='#64748B'),
            linecolor='#E2E8F0',
            showline=True,
            tickangle=45 if len(daily_pnl_df_sorted) > 5 else 0,
            tickmode='array',
            tickvals=daily_pnl_df_sorted['Date_Str'].tolist(),
            ticktext=daily_pnl_df_sorted['Date_Str'].tolist()
        ),
        yaxis=dict(
            title=f"Daily Net P&L ({currency_symbol})",
            side="left",
            showgrid=True,
            gridcolor='#F1F5F9',
            gridwidth=1,
            tickprefix=currency_symbol,
            tickformat=',.0f',
            tickfont=dict(size=10, color='#64748B'),
            linecolor='#E2E8F0',
            showline=True
        ),
        yaxis2=dict(
            title=f"Capital ({currency_symbol})",
            side="right",
            overlaying="y",
            showgrid=False,
            tickprefix=currency_symbol,
            tickformat=',.0f',
            tickfont=dict(size=10, color='#FFA500'),
            linecolor='#FFA500',
            showline=True
        )
    )
    
    return fig

def create_intraday_dashboard(data_dict, live_pnl_df, region="INDIA"):
    """Create intraday dashboard for either INDIA or GLOBAL region"""
    open_df = data_dict['open_positions']
    closed_df = data_dict['closed_positions']
    summary = data_dict['summary']
    
    if open_df.empty and closed_df.empty:
        st.info(f"📭 NO ACTIVE POSITIONS.")
        return
    
    format_currency_func = get_currency_formatter(region)
    currency_symbol = get_currency_symbol(region)
    
    # Display total P&L
    total_pnl = summary.get('total_pnl', 0)
    pnl_color = "green" if total_pnl > 0 else "red" if total_pnl < 0 else "gray"
    
    st.markdown(
        f"""
        <div style="text-align: center; margin-bottom: 1.2rem;">
            <span style="font-size: 2.4rem; font-weight: 800; color: {pnl_color};">
                {format_currency_func(total_pnl)}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Display key metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(create_metric_card("Closed P&L", format_currency_func(summary.get('total_closed_pnl', 0)), "#1f77b4"), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric_card("Unrealized P&L", format_currency_func(summary.get('total_unrealized_pnl', 0)), "#ff7f0e"), unsafe_allow_html=True)
    with col3:
        st.markdown(create_metric_card("Traded Volume", format_currency_func(summary.get('total_traded_volume', 0)), "#2ca02c"), unsafe_allow_html=True)
    with col4:
        st.markdown(create_metric_card("Open Exposure", format_currency_func(summary.get('total_open_exposure', 0)), "#d62728"), unsafe_allow_html=True)
    with col5:
        st.markdown(create_metric_card("Open Positions", summary.get('open_positions_count', 0), "#9467bd"), unsafe_allow_html=True)
    with col6:
        st.markdown(create_metric_card("Closed Positions", summary.get('closed_positions_count', 0), "#8c564b"), unsafe_allow_html=True)
    
    # Show last updated time
    if not live_pnl_df.empty and 'DateTime' in live_pnl_df.columns:
        last_datetime = live_pnl_df['DateTime'].iloc[-1]
        timezone_str = "IST" if region == "INDIA" else "ET"
        formatted_time = last_datetime.strftime(f'%Y-%m-%d %H:%M:%S {timezone_str}')
        st.caption(f"📊 Last Updated: {formatted_time}")
    
    # Display live P&L chart
    if not live_pnl_df.empty:
        st.divider()
        fig = create_live_pnl_chart(live_pnl_df, currency_symbol)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # Display open positions
    if not open_df.empty:
        st.divider()
        st.subheader("📈 Open Positions")
        
        open_display_df = open_df[[
            'tradingsymbol', 'position_type', 'net_quantity',
            'avg_price', 'last_price', 'unrealized_pnl', 'open_exposure'
        ]].copy()
        
        open_display_df = open_display_df.rename(columns={
            'tradingsymbol': 'Symbol',
            'position_type': 'Position',
            'net_quantity': 'Quantity',
            'avg_price': 'Avg Price',
            'last_price': 'Last Price',
            'unrealized_pnl': 'Unrealized P&L',
            'open_exposure': 'Open Exposure'
        })
        
        # Format columns
        for col in ['Avg Price', 'Last Price', 'Unrealized P&L', 'Open Exposure']:
            open_display_df[col] = open_display_df[col].apply(format_currency_func)
        
        table_html = create_html_table(
            open_display_df,
            ['Symbol', 'Position', 'Quantity', 'Avg Price', 'Last Price', 'Unrealized P&L', 'Open Exposure'],
            currency_symbol
        )
        st.markdown(table_html, unsafe_allow_html=True)
    
    # Display closed positions
    if not closed_df.empty:
        st.divider()
        st.subheader("📊 Closed Positions (Today)")
        
        # Sort by P&L BEFORE any processing
        closed_df_sorted = closed_df.sort_values(by='pnl', ascending=False)
        
        closed_display_df = closed_df_sorted[[
            'tradingsymbol', 'buy_quantity', 'buy_price',
            'sell_quantity', 'sell_price', 'pnl'
        ]].copy()
        
        closed_display_df = closed_display_df.rename(columns={
            'tradingsymbol': 'Symbol',
            'buy_quantity': 'Buy Qty',
            'buy_price': 'Buy Price',
            'sell_quantity': 'Sell Qty',
            'sell_price': 'Sell Price',
            'pnl': 'Realized P&L'
        })
        
        # Format columns AFTER sorting
        for col in ['Buy Price', 'Sell Price', 'Realized P&L']:
            closed_display_df[col] = closed_display_df[col].apply(format_currency_func)
        
        table_html = create_html_table(
            closed_display_df,
            ['Symbol', 'Buy Qty', 'Buy Price', 'Sell Qty', 'Sell Price', 'Realized P&L'],
            currency_symbol
        )
        st.markdown(table_html, unsafe_allow_html=True)

def create_daily_pnl_dashboard(daily_pnl_df, region="INDIA"):
    """Create Daily PnL dashboard"""
    if daily_pnl_df.empty:
        st.info(f"📭 No Daily PnL data available for {region}")
        return
    
    format_currency_func = get_currency_formatter(region)
    currency_symbol = get_currency_symbol(region)
    daily_pnl_sorted = daily_pnl_df.sort_values('Date', ascending=True)
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_gross = daily_pnl_df['Gross P&L'].sum()
        gross_color = "#2ca02c" if total_gross >= 0 else "#d62728"
        st.markdown(create_metric_card("Total Gross P&L", format_currency_func(total_gross), gross_color), unsafe_allow_html=True)
    
    with col2:
        total_charges = daily_pnl_df['Charges'].sum()
        st.markdown(create_metric_card("Total Charges", format_currency_func(total_charges), "#d62728"), unsafe_allow_html=True)
    
    with col3:
        total_net = daily_pnl_df['Net P&L'].sum()
        net_color = "#10B981" if total_net >= 0 else "#EF4444"
        st.markdown(create_metric_card("Total Net P&L", format_currency_func(total_net), net_color), unsafe_allow_html=True)
    
    with col4:
        if not daily_pnl_sorted.empty and 'Capital' in daily_pnl_sorted.columns:
            current_capital = daily_pnl_sorted['Capital'].iloc[-1]
            st.markdown(create_metric_card("Current Capital", format_currency_func(current_capital), "#FFA500"), unsafe_allow_html=True)
        else:
            st.markdown(create_metric_card("Current Capital", "N/A", "#FFA500"), unsafe_allow_html=True)
    
    st.divider()
    
    # Display daily P&L chart
    if 'Capital' in daily_pnl_df.columns and not daily_pnl_df['Capital'].isnull().all():
        fig = create_daily_pnl_chart(daily_pnl_df, currency_symbol)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

def create_refresh_button(key_suffix):
    """Create a refresh button that clears cache"""
    if st.button("🔄 Refresh Data", type="secondary", key=f"refresh_{key_suffix}"):
        st.cache_data.clear()
        st.rerun()

# ===================================================================
# 🎨 CSS for Bigger Tabs - UPDATED FOR 5 TABS
# ===================================================================
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        padding-top: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        font-size: 16px;
        font-weight: 600;
        padding: 8px 20px;
        border-radius: 6px 6px 0 0;
        background-color: #f0f2f6;
        white-space: nowrap;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        border-bottom: 3px solid #FF4B4B;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1px solid #e0e0e0;
        flex-wrap: wrap;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1rem;
    }
    
    .main .block-container {
        padding-top: 1rem;
    }
    
    /* News terminal styling */
    .news-terminal {
        font-family: 'Courier New', monospace;
        background-color: #000000;
        color: #00FF00;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #00FF00;
        max-height: 600px;
        overflow-y: auto;
        line-height: 1.6;
    }
    
    .news-item {
        margin-bottom: 15px;
        border-left: 3px solid;
        padding-left: 10px;
    }
    
    .news-timestamp {
        color: #888888;
        font-size: 12px;
        margin-bottom: 4px;
    }
    
    .news-content {
        color: #FFFFFF;
        font-size: 14px;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

# ===================================================================
# 📥 Load Data (Keep all your existing data loading)
# ===================================================================
df_india_raw = load_sheet_data(sheet_gid="649765105")
df_india_live_pnl_raw = load_sheet_data(sheet_gid="1065660372")
df_india_daily_pnl_raw = load_sheet_data(sheet_gid="795838620")

df_global_raw = load_sheet_data(sheet_gid="94252270")
df_global_live_pnl_raw = load_sheet_data(sheet_gid="1297846329")
df_global_daily_pnl_raw = load_sheet_data(sheet_gid="563240267")

# Process data
india_data = process_india_data(df_india_raw)
india_live_pnl_data = process_live_pnl_data(df_india_live_pnl_raw)
india_daily_pnl_data = process_daily_pnl_data(df_india_daily_pnl_raw, region="INDIA")

global_data = process_india_data(df_global_raw) if not df_global_raw.empty else {
    'open_positions': pd.DataFrame(), 'closed_positions': pd.DataFrame(), 'summary': {}
}
global_live_pnl_data = process_live_pnl_data(df_global_live_pnl_raw)
global_daily_pnl_data = process_daily_pnl_data(df_global_daily_pnl_raw, region="GLOBAL")

# ===================================================================
# 📊 Create Tabs - UPDATED WITH NEWS TAB
# ===================================================================
# Create 5 tabs including the new News tab
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌍 **GLOBAL (INTRA)**", 
    "📊 **GLOBAL (DAILY)**",
    "🇮🇳 **INDIA (INTRA)**",
    "📊 **INDIA (DAILY)**",
    "📰 **NEWS & SENTIMENT**"  # New tab
])

with tab1:
    # Add refresh button at top-right
    col1, col2 = st.columns([5, 1])
    with col2:
        create_refresh_button("global_intra")
    
    create_intraday_dashboard(global_data, global_live_pnl_data, region="GLOBAL")

with tab2:
    col1, col2 = st.columns([5, 1])
    with col2:
        create_refresh_button("global_daily")
    
    create_daily_pnl_dashboard(global_daily_pnl_data, region="GLOBAL")

with tab3:
    col1, col2 = st.columns([5, 1])
    with col2:
        create_refresh_button("india_intra")
    
    create_intraday_dashboard(india_data, india_live_pnl_data, region="INDIA")

with tab4:
    col1, col2 = st.columns([5, 1])
    with col2:
        create_refresh_button("india_daily")
    
    create_daily_pnl_dashboard(india_daily_pnl_data, region="INDIA")

with tab5:
    # NEWS & SENTIMENT TAB
    col1, col2 = st.columns([5, 1])
    with col2:
        create_refresh_button("news_sentiment")
    
    # Check if news sheet URL is configured
    if not NEWS_SHEET_URL:
        st.warning("""
        ⚠️ News not Updated.
        """)
    else:
        # Initialize and display news sentiment dashboard
        analyzer = NewsSentimentAnalyzer(google_sheet_url=NEWS_SHEET_URL)
        analyzer.display_dashboard()
