import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, date
import pytz

# ===================================================================
# 🛠️ CONFIGURATION
# ===================================================================
REFRESH_INTERVAL_SEC = 10  # Data refreshes every 10 seconds

# ===================================================================
# 🔐 Load Google Sheet URL from Streamlit Secrets
# ===================================================================
try:
    GOOGLE_SHEET_CSV_URL = st.secrets["google_sheet"]["csv_url"]
except KeyError:
    st.error("🔐 Missing Google Sheet URL in Streamlit Secrets.")
    st.stop()

st.set_page_config(
    page_title="BITQCODE Dashboard",
    page_icon="💼",
    layout="wide"
)

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
        # IST timezone (UTC+5:30)
        tz = pytz.timezone('Asia/Kolkata')
        now = datetime.now(tz)
        return now.strftime('%Y-%m-%d %H:%M:%S IST')
    else:  # GLOBAL - Use US/Eastern or UTC
        # Using US Eastern Time (ET)
        tz = pytz.timezone('US/Eastern')
        now = datetime.now(tz)
        return now.strftime('%Y-%m-%d %H:%M:%S ET')

# ===================================================================
# 📥 Load & Clean Data — WITH AUTOMATIC REFRESH
# ===================================================================
@st.cache_data(ttl=REFRESH_INTERVAL_SEC, show_spinner=False)
def load_sheet_data(sheet_gid="0"):
    """Load specific sheet from Google Sheets using gid parameter"""
    try:
        # Construct URL with gid parameter for specific sheet
        if "export?format=csv" in GOOGLE_SHEET_CSV_URL:
            # Replace or add gid parameter
            if "gid=" in GOOGLE_SHEET_CSV_URL:
                url = GOOGLE_SHEET_CSV_URL.split("&gid=")[0] + f"&gid={sheet_gid}"
            else:
                url = GOOGLE_SHEET_CSV_URL + f"&gid={sheet_gid}"
        else:
            url = GOOGLE_SHEET_CSV_URL + f"?gid={sheet_gid}&format=csv"
        
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"❌ Failed to load sheet {sheet_gid}: {str(e)[:150]}...")
        return pd.DataFrame()

@st.cache_data(ttl=REFRESH_INTERVAL_SEC)
def process_live_pnl_data(df_raw):
    """Process Live PnL data - filter for today's date only"""
    if df_raw.empty:
        return pd.DataFrame()
    
    # Clean column names
    df = df_raw.copy()
    df.columns = df.columns.str.strip()
    
    # Check for required columns
    required_cols = ['DateTime', 'Total PnL']
    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame()
    
    # Convert DateTime to datetime and Total PnL to numeric
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    df['Total PnL'] = pd.to_numeric(df['Total PnL'], errors='coerce')
    
    # Drop rows with invalid dates or PnL
    df = df.dropna(subset=['DateTime', 'Total PnL'])
    
    if df.empty:
        return df
    
    # # Get today's date in IST (Asia/Kolkata)
    # ist_tz = pytz.timezone('Asia/Kolkata')
    # today_ist = datetime.now(ist_tz).date()
    
    # # Filter for today's data only (based on date part, ignoring time)
    # df['Date'] = df['DateTime'].dt.date
    # df_today = df[df['Date'] == today_ist].copy()

    # Get the latest date from the data
    df['Date'] = df['DateTime'].dt.date
    latest_date = df['Date'].max()
    
    # Filter for the latest date's data only
    df_today = df[df['Date'] == latest_date].copy()
    
    # Sort by DateTime
    df_today = df_today.sort_values('DateTime')
    
    return df_today

@st.cache_data(ttl=REFRESH_INTERVAL_SEC)
def process_global_data(df_raw):
    """Process GLOBAL data with caching"""
    if df_raw.empty:
        return pd.DataFrame()
    
    # Check for required columns
    required_cols = {
        'Strategy Name', 'Account', 'Symbol', 'SecType',
        'Currency', 'Position', 'AvgCost', 'MarketPrice'
    }
    
    # Check which required columns exist
    missing_cols = required_cols - set(df_raw.columns)
    if missing_cols:
        return pd.DataFrame()  # Silent fail - will be handled in dashboard
    
    # Select only needed columns
    df = df_raw[list(required_cols)].copy()
    
    # Convert to numeric
    for col in ['Position', 'AvgCost', 'MarketPrice']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 🔥 ULTRA-AGGRESSIVE CLEANING
    df = df.dropna(subset=['Strategy Name', 'Symbol', 'SecType', 'Position', 'AvgCost', 'MarketPrice'])
    
    # Remove rows with empty strings or whitespace only
    text_columns = ['Strategy Name', 'Symbol', 'SecType', 'Account']
    for col in text_columns:
        if col in df.columns:
            df = df[df[col].astype(str).str.strip() != '']
            df = df[df[col].astype(str).str.strip() != 'nan']
            df = df[df[col].notna()]
    
    # Remove zero positions and invalid numeric values
    df = df[df['Position'] != 0]
    df = df[df['AvgCost'] > 0]
    df = df[df['MarketPrice'] > 0]
    
    # Reset index after all filtering
    df = df.reset_index(drop=True)
    
    if df.empty:
        return df
    
    # ===================================================================
    # 📊 Compute P&L
    # ===================================================================
    def calculate_pnl(row):
        qty = row['Position']
        avg = row['AvgCost']
        mp = row['MarketPrice']
        return (mp - avg) * qty if qty > 0 else (avg - mp) * abs(qty)
    
    df['UnrealizedPnL'] = df.apply(calculate_pnl, axis=1)
    df['CostBasis'] = df['Position'].abs() * df['AvgCost']
    df['UnrealizedPnL%'] = np.where(
        df['CostBasis'] != 0,
        (df['UnrealizedPnL'] / df['CostBasis']) * 100,
        0
    )
    df['Long/Short'] = df['Position'].apply(lambda x: 'Long' if x > 0 else 'Short')
    df['MarketValue'] = df['Position'] * df['MarketPrice']
    
    df['Account'] = df['Account'].apply(mask_account)
    df = df.iloc[df['UnrealizedPnL'].abs().argsort()[::-1]].reset_index(drop=True)
    
    return df

@st.cache_data(ttl=REFRESH_INTERVAL_SEC)
def process_india_data(df_raw):
    """Process INDIA data with the new format"""
    if df_raw.empty:
        return {
            'open_positions': pd.DataFrame(),
            'closed_positions': pd.DataFrame(),
            'summary': {}
        }
    
    # Create a copy to avoid modifying the original
    df = df_raw.copy()
    
    # Clean column names by stripping whitespace
    df.columns = df.columns.str.strip()
    
    # Define expected columns for INDIA format
    expected_cols = [
        's_no', 'tradingsymbol', 'buy_value', 'buy_price', 
        'buy_quantity', 'sell_quantity', 'sell_price', 
        'sell_value', 'last_price', 'pnl'
    ]
    
    # Check if we have the expected columns
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        st.warning(f"Missing expected columns in INDIA data: {missing_cols}")
        return {
            'open_positions': pd.DataFrame(),
            'closed_positions': pd.DataFrame(),
            'summary': {}
        }
    
    # Convert numeric columns
    numeric_cols = ['buy_value', 'buy_price', 'buy_quantity', 'sell_quantity', 
                   'sell_price', 'sell_value', 'last_price', 'pnl']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Clean data
    df = df.dropna(subset=['tradingsymbol'])
    df = df[df['tradingsymbol'].astype(str).str.strip() != '']
    
    # ===================================================================
    # SEPARATE OPEN AND CLOSED POSITIONS
    # ===================================================================
    
    # Closed positions: buy_quantity == sell_quantity (fully closed)
    closed_mask = (df['buy_quantity'] > 0) & (df['sell_quantity'] > 0) & (df['buy_quantity'] == df['sell_quantity'])
    closed_df = df[closed_mask].copy()
    
    # Open positions: buy_quantity != sell_quantity or sell_quantity == 0
    open_mask = ~closed_mask
    open_df = df[open_mask].copy()
    
    # Calculate additional metrics for open positions
    if not open_df.empty:
        # Calculate net quantity
        open_df['net_quantity'] = open_df['buy_quantity'] - open_df['sell_quantity']
        
        # Calculate average price for open positions
        open_df['avg_price'] = np.where(
            open_df['net_quantity'] != 0,
            (open_df['buy_value'] - open_df['sell_value']) / open_df['net_quantity'],
            0
        )
        
        # Calculate unrealized P&L for open positions
        open_df['unrealized_pnl'] = (open_df['last_price'] - open_df['avg_price']) * open_df['net_quantity']
        
        # Calculate open exposure
        open_df['open_exposure'] = open_df['net_quantity'] * open_df['last_price']
        
        # Determine position type
        open_df['position_type'] = open_df['net_quantity'].apply(lambda x: 'Long' if x > 0 else 'Short' if x < 0 else 'Flat')
        
        # Sort by unrealized P&L
        open_df = open_df.sort_values('unrealized_pnl', ascending=False)
    
    # Calculate summary metrics
    total_traded_volume = df['buy_value'].sum() + df['sell_value'].sum()
    total_closed_pnl = closed_df['pnl'].sum() if not closed_df.empty else 0
    total_unrealized_pnl = open_df['unrealized_pnl'].sum() if not open_df.empty else 0
    total_open_exposure = open_df['open_exposure'].abs().sum() if not open_df.empty else 0
    
    # Calculate number of positions
    open_positions_count = len(open_df)
    closed_positions_count = len(closed_df)
    
    return {
        'open_positions': open_df,
        'closed_positions': closed_df,
        'summary': {
            'total_traded_volume': total_traded_volume,
            'total_closed_pnl': total_closed_pnl,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_open_exposure': total_open_exposure,
            'open_positions_count': open_positions_count,
            'closed_positions_count': closed_positions_count,
            'total_pnl': total_closed_pnl + total_unrealized_pnl
        }
    }

def create_dashboard(data_dict, live_pnl_df, region="INDIA"):
    """Create dashboard for either INDIA or GLOBAL region"""
    
    open_df = data_dict['open_positions']
    closed_df = data_dict['closed_positions']
    summary = data_dict['summary']
    
    if open_df.empty and closed_df.empty:
        st.info(f"📭 NO ACTIVE POSITIONS.")
        return
    
    # Get appropriate currency formatter
    format_currency_func = format_inr if region == "INDIA" else lambda x: format_currency(x, "$")
    currency_symbol = "₹" if region == "INDIA" else "$"
    
    # ===================================================================
    # 🎯 BIG BOLD TOTAL P&L (Closed + Unrealized)
    # ===================================================================
    total_pnl = summary.get('total_pnl', 0)
    capital = 991002.7 if region == "INDIA" else 1000000  # Adjust capital as needed
    
    # Determine arrow and color
    if total_pnl > 0:
        pnl_color = "green"
        pnl_symbol = "▲"
    elif total_pnl < 0:
        pnl_color = "red"
        pnl_symbol = "▼"
    else:
        pnl_color = "gray"
    
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
    
    # ===================================================================
    # 📊 KEY METRICS
    # ===================================================================
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.85rem; font-weight: 600; color: #1f77b4; margin-bottom: 0.2rem;">Closed P&L</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #1f77b4;">{format_currency_func(summary.get('total_closed_pnl', 0))}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.85rem; font-weight: 600; color: #ff7f0e; margin-bottom: 0.2rem;">Unrealized P&L</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #ff7f0e;">{format_currency_func(summary.get('total_unrealized_pnl', 0))}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.85rem; font-weight: 600; color: #2ca02c; margin-bottom: 0.2rem;">Traded Volume</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #2ca02c;">{format_currency_func(summary.get('total_traded_volume', 0))}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.85rem; font-weight: 600; color: #d62728; margin-bottom: 0.2rem;">Open Exposure</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #d62728;">{format_currency_func(summary.get('total_open_exposure', 0))}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col5:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.85rem; font-weight: 600; color: #9467bd; margin-bottom: 0.2rem;">Open Positions</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #9467bd;">{summary.get('open_positions_count', 0)}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col6:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.85rem; font-weight: 600; color: #8c564b; margin-bottom: 0.2rem;">Closed Positions</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #8c564b;">{summary.get('closed_positions_count', 0)}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Show last updated datetime from LIVE PnL sheet
    if not live_pnl_df.empty and 'DateTime' in live_pnl_df.columns:
        last_datetime = live_pnl_df['DateTime'].iloc[-1]
        
        # Format the datetime nicely
        if isinstance(last_datetime, pd.Timestamp):
            # Format as simple date-time
            timezone_str = "IST" if region == "INDIA" else "ET"

            formatted_time = last_datetime.strftime(f'%Y-%m-%d %H:%M:%S {timezone_str}')
        else:
            formatted_time = str(last_datetime)
        
        st.caption(f"📊 Last Updated: {formatted_time}")
    
    
    # ===================================================================
    # 📈 TODAY'S LIVE P&L CHART - SINGLE LINE WITH COLOR BY VALUE & EXTREMES
    # ===================================================================
    if not live_pnl_df.empty:
        st.divider()
        
        # Sort by DateTime
        live_pnl_df_sorted = live_pnl_df.sort_values('DateTime')
        
        # Find first occurrence of highest and lowest values
        highest_value = live_pnl_df_sorted['Total PnL'].max()
        lowest_value = live_pnl_df_sorted['Total PnL'].min()
        
        # Get first occurrence of highest and lowest
        highest_row = live_pnl_df_sorted[live_pnl_df_sorted['Total PnL'] == highest_value].iloc[0]
        lowest_row = live_pnl_df_sorted[live_pnl_df_sorted['Total PnL'] == lowest_value].iloc[0]
        
        # Create the chart
        fig = go.Figure()
        
        # Create a single line that changes color based on value
        segments = []
        current_segment = {'x': [], 'y': [], 'color': None}
        
        for i in range(len(live_pnl_df_sorted)):
            current_val = live_pnl_df_sorted['Total PnL'].iloc[i]
            current_time = live_pnl_df_sorted['DateTime'].iloc[i]
            current_color = '#10B981' if current_val >= 0 else '#EF4444'  # Green/Red
            
            if not current_segment['x']:
                # First point
                current_segment['x'].append(current_time)
                current_segment['y'].append(current_val)
                current_segment['color'] = current_color
            elif current_segment['color'] == current_color:
                # Same color, continue segment
                current_segment['x'].append(current_time)
                current_segment['y'].append(current_val)
            else:
                # Color changed (crossed zero), handle the transition
                prev_val = live_pnl_df_sorted['Total PnL'].iloc[i-1]
                prev_time = live_pnl_df_sorted['DateTime'].iloc[i-1]
                
                # Calculate the exact zero crossing point
                m = (current_val - prev_val) / ((current_time - prev_time).total_seconds())
                zero_time_seconds = -prev_val / m if m != 0 else 0
                zero_time = prev_time + pd.Timedelta(seconds=zero_time_seconds)
                
                # Add point at zero to complete current segment
                current_segment['x'].append(zero_time)
                current_segment['y'].append(0)
                segments.append(current_segment.copy())
                
                # Start new segment from zero point
                current_segment = {
                    'x': [zero_time, current_time],
                    'y': [0, current_val],
                    'color': current_color
                }
        
        # Add the last segment
        if current_segment['x']:
            segments.append(current_segment)
        
        # Add all segments as separate traces
        for segment in segments:
            fig.add_trace(go.Scatter(
                x=segment['x'],
                y=segment['y'],
                mode='lines',
                line=dict(
                    shape='spline',
                    smoothing=1.0,
                    width=3,
                    color=segment['color']
                ),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add invisible trace for hover information
        fig.add_trace(go.Scatter(
            x=live_pnl_df_sorted['DateTime'],
            y=live_pnl_df_sorted['Total PnL'],
            mode='lines',
            line=dict(width=0),  # Invisible
            hovertemplate=f'<b>%{{x|%H:%M:%S}}</b><br>{currency_symbol}%{{y:,.2f}}<extra></extra>',
            showlegend=False,
            name='Live P&L'
        ))
        
        # Add zero line
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="#94A3B8",
            line_width=1,
            opacity=0.3
        )
        
        # Add area fill with appropriate color
        x_full = live_pnl_df_sorted['DateTime'].tolist()
        y_full = live_pnl_df_sorted['Total PnL'].tolist()
        
        # Fill area
        fig.add_trace(go.Scatter(
            x=x_full,
            y=y_full,
            mode='none',
            fill='tozeroy',
            fillcolor='rgba(16, 185, 129, 0.1)',  # Light green for positive
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Also fill negative area separately (overlay)
        fig.add_trace(go.Scatter(
            x=x_full,
            y=[min(y, 0) for y in y_full],  # Only negative part
            mode='none',
            fill='tozeroy',
            fillcolor='rgba(239, 68, 68, 0.1)',  # Light red for negative
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # ===================================================================
        # 🎯 ADD HIGHEST AND LOWEST POINT MARKERS
        # ===================================================================
        
        # Mark highest point (first occurrence)
        fig.add_trace(go.Scatter(
            x=[highest_row['DateTime']],
            y=[highest_value],
            mode='markers+text',
            marker=dict(
                size=12,
                color='#10B981',
                symbol='triangle-up',
                line=dict(width=2, color='white')
            ),
            text=[f"  High: {currency_symbol}{highest_value:,.0f}"],
            textposition="top center",
            textfont=dict(size=11, color='#10B981', family='Arial'),
            hovertemplate=f'<b>Highest: {currency_symbol}{highest_value:,.2f}</b><br>Time: %{{x|%H:%M:%S}}<extra></extra>',
            showlegend=False,
            name='Highest'
        ))
        
        # Mark lowest point (first occurrence)
        fig.add_trace(go.Scatter(
            x=[lowest_row['DateTime']],
            y=[lowest_value],
            mode='markers+text',
            marker=dict(
                size=12,
                color='#EF4444',
                symbol='triangle-down',
                line=dict(width=2, color='white')
            ),
            text=[f"  Low: {currency_symbol}{lowest_value:,.0f}"],
            textposition="bottom center",
            textfont=dict(size=11, color='#EF4444', family='Arial'),
            hovertemplate=f'<b>Lowest: {currency_symbol}{lowest_value:,.2f}</b><br>Time: %{{x|%H:%M:%S}}<extra></extra>',
            showlegend=False,
            name='Lowest'
        ))
        
        # Clean layout
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
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # # Show latest value with more info
        # if len(live_pnl_df_sorted) > 0:
        #     latest_pnl = live_pnl_df_sorted['Total PnL'].iloc[-1]
        #     latest_time = live_pnl_df_sorted['DateTime'].iloc[-1].strftime('%H:%M:%S')
        #     pnl_color = "#10B981" if latest_pnl >= 0 else "#EF4444"
            
        #     st.markdown(
        #         f"""
        #         <div style="text-align: center; margin-top: 10px; padding: 10px; background: {pnl_color}10; border-radius: 8px;">
        #             <span style="font-size: 1.1rem; font-weight: 600; color: {pnl_color};">
        #                 📊 Latest: {format_currency_func(latest_pnl)} at {latest_time}
        #             </span>
        #             <br>
        #             <span style="font-size: 0.9rem; color: #64748B;">
        #                 Today's High: {format_currency_func(highest_value)} • Today's Low: {format_currency_func(lowest_value)}
        #             </span>
        #         </div>
        #         """,
        #         unsafe_allow_html=True
        #     )
    
    # ===================================================================
    # 📋 OPEN POSITIONS
    # ===================================================================
    if not open_df.empty:
        st.divider()
        st.subheader("📈 Open Positions")
        
        # Prepare display dataframe for open positions
        open_display_df = open_df[[
            'tradingsymbol', 'position_type', 'net_quantity',
            'avg_price', 'last_price', 'unrealized_pnl', 'open_exposure'
        ]].copy()
        
        # Rename columns for better display
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
        open_display_df['Avg Price'] = open_display_df['Avg Price'].apply(format_currency_func)
        open_display_df['Last Price'] = open_display_df['Last Price'].apply(format_currency_func)
        open_display_df['Unrealized P&L'] = open_display_df['Unrealized P&L'].apply(format_currency_func)
        open_display_df['Open Exposure'] = open_display_df['Open Exposure'].apply(format_currency_func)
        
        # Create HTML table for open positions
        html_table_open = f"""
        <div style="overflow-x: auto;">
        <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
            <thead>
                <tr style="background-color: #f2f2f2;">
                    <th style="padding: 10px; text-align: left; border-bottom: 1px solid #ddd;">Symbol</th>
                    <th style="padding: 10px; text-align: left; border-bottom: 1px solid #ddd;">Position</th>
                    <th style="padding: 10px; text-align: right; border-bottom: 1px solid #ddd;">Quantity</th>
                    <th style="padding: 10px; text-align: right; border-bottom: 1px solid #ddd;">Avg Price</th>
                    <th style="padding: 10px; text-align: right; border-bottom: 1px solid #ddd;">Last Price</th>
                    <th style="padding: 10px; text-align: right; border-bottom: 1px solid #ddd;">Unrealized P&L</th>
                    <th style="padding: 10px; text-align: right; border-bottom: 1px solid #ddd;">Open Exposure</th>
                </tr>
            </thead>
            <tbody>
        """
        
        # Add rows with color coding for P&L
        for _, row in open_display_df.iterrows():
            html_table_open += '<tr>'
            
            # Symbol
            html_table_open += f'<td style="padding: 8px; border-bottom: 1px solid #ddd;">{row["Symbol"]}</td>'
            
            # Position
            html_table_open += f'<td style="padding: 8px; border-bottom: 1px solid #ddd;">{row["Position"]}</td>'
            
            # Quantity
            html_table_open += f'<td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{row["Quantity"]}</td>'
            
            # Avg Price
            html_table_open += f'<td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{row["Avg Price"]}</td>'
            
            # Last Price
            html_table_open += f'<td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{row["Last Price"]}</td>'
            
            # Unrealized P&L - with color coding
            pnl_style = "padding: 8px; border-bottom: 1px solid #ddd; text-align: right; font-weight: bold;"
            pnl_value = row["Unrealized P&L"]
            if f"{currency_symbol}-" in str(pnl_value) or "-" in str(pnl_value) or "−" in str(pnl_value):
                pnl_style += "color: red;"
            else:
                pnl_style += "color: green;"
            html_table_open += f'<td style="{pnl_style}">{pnl_value}</td>'
            
            # Open Exposure
            html_table_open += f'<td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{row["Open Exposure"]}</td>'
            
            html_table_open += '</tr>'
        
        html_table_open += """
            </tbody>
        </table>
        </div>
        """
        
        st.markdown(html_table_open, unsafe_allow_html=True)
    
    # ===================================================================
    # 📋 CLOSED POSITIONS
    # ===================================================================
    if not closed_df.empty:
        st.divider()
        st.subheader("📊 Closed Positions (Today)")
        
        # Prepare display dataframe for closed positions
        closed_display_df = closed_df[[
            'tradingsymbol', 'buy_quantity', 'buy_price',
            'sell_quantity', 'sell_price', 'pnl'
        ]].copy()
        
        # Rename columns for better display
        closed_display_df = closed_display_df.rename(columns={
            'tradingsymbol': 'Symbol',
            'buy_quantity': 'Buy Qty',
            'buy_price': 'Buy Price',
            'sell_quantity': 'Sell Qty',
            'sell_price': 'Sell Price',
            'pnl': 'Realized P&L'
        })
        
        # Format columns
        closed_display_df['Buy Price'] = closed_display_df['Buy Price'].apply(format_currency_func)
        closed_display_df['Sell Price'] = closed_display_df['Sell Price'].apply(format_currency_func)
        closed_display_df['Realized P&L'] = closed_display_df['Realized P&L'].apply(format_currency_func)
        
        # Create HTML table for closed positions
        html_table_closed = f"""
        <div style="overflow-x: auto;">
        <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
            <thead>
                <tr style="background-color: #f2f2f2;">
                    <th style="padding: 10px; text-align: left; border-bottom: 1px solid #ddd;">Symbol</th>
                    <th style="padding: 10px; text-align: right; border-bottom: 1px solid #ddd;">Buy Qty</th>
                    <th style="padding: 10px; text-align: right; border-bottom: 1px solid #ddd;">Buy Price</th>
                    <th style="padding: 10px; text-align: right; border-bottom: 1px solid #ddd;">Sell Qty</th>
                    <th style="padding: 10px; text-align: right; border-bottom: 1px solid #ddd;">Sell Price</th>
                    <th style="padding: 10px; text-align: right; border-bottom: 1px solid #ddd;">Realized P&L</th>
                </tr>
            </thead>
            <tbody>
        """
        
        # Add rows with color coding for P&L
        for _, row in closed_display_df.iterrows():
            html_table_closed += '<tr>'
            
            # Symbol
            html_table_closed += f'<td style="padding: 8px; border-bottom: 1px solid #ddd;">{row["Symbol"]}</td>'
            
            # Buy Qty
            html_table_closed += f'<td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{row["Buy Qty"]}</td>'
            
            # Buy Price
            html_table_closed += f'<td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{row["Buy Price"]}</td>'
            
            # Sell Qty
            html_table_closed += f'<td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{row["Sell Qty"]}</td>'
            
            # Sell Price
            html_table_closed += f'<td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{row["Sell Price"]}</td>'
            
            # Realized P&L - with color coding
            pnl_style = "padding: 8px; border-bottom: 1px solid #ddd; text-align: right; font-weight: bold;"
            pnl_value = row["Realized P&L"]
            if f"{currency_symbol}-" in str(pnl_value) or "-" in str(pnl_value) or "−" in str(pnl_value):
                pnl_style += "color: red;"
            else:
                pnl_style += "color: green;"
            html_table_closed += f'<td style="{pnl_style}">{pnl_value}</td>'
            
            html_table_closed += '</tr>'
        
        html_table_closed += """
            </tbody>
        </table>
        </div>
        """
        
        st.markdown(html_table_closed, unsafe_allow_html=True)
    
    # ===================================================================
    # 📈 OTHER CHARTS FOR INDIA
    # ===================================================================
    if not open_df.empty or not closed_df.empty:
        st.divider()
        st.subheader("📊 Performance Analysis")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # P&L Distribution Chart
            st.subheader("💰 P&L Distribution")
            
            if not open_df.empty and not closed_df.empty:
                # Combine open and closed P&L data
                pnl_data = pd.DataFrame({
                    'Category': ['Closed P&L', 'Unrealized P&L'],
                    'Amount': [summary['total_closed_pnl'], summary['total_unrealized_pnl']]
                })
                
                fig1 = px.bar(
                    pnl_data,
                    x='Category',
                    y='Amount',
                    color='Category',
                    color_discrete_map={'Closed P&L': '#1f77b4', 'Unrealized P&L': '#ff7f0e'},
                    text=[format_inr(x) for x in pnl_data['Amount']]
                )
                fig1.update_layout(
                    height=300,
                    showlegend=False,
                    xaxis_title="",
                    yaxis_title="Amount (₹)",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.info("No P&L data available for chart.")
        
        with chart_col2:
            # Positions Overview Chart
            st.subheader("📊 Positions Overview")
            
            positions_data = pd.DataFrame({
                'Category': ['Open Positions', 'Closed Positions'],
                'Count': [summary['open_positions_count'], summary['closed_positions_count']]
            })
            
            fig2 = px.pie(
                positions_data,
                values='Count',
                names='Category',
                hole=0.4,
                color_discrete_sequence=['#9467bd', '#8c564b']
            )
            fig2.update_layout(
                height=300,
                showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5, x=1.1)
            )
            st.plotly_chart(fig2, use_container_width=True)

@st.cache_data(ttl=REFRESH_INTERVAL_SEC)
def process_daily_pnl_data(df_raw, region="INDIA"):
    """Process Daily PnL data for INDIA and GLOBAL"""
    if df_raw.empty:
        return pd.DataFrame()
    
    # Clean column names
    df = df_raw.copy()
    df.columns = df.columns.str.strip()
    
    # Check for required columns
    required_cols = ['Date', 'Gross P&L', 'Charges', 'Net P&L']
    if not all(col in df.columns for col in required_cols):
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
    
    # Verify we have the required columns
    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame()
    
    # Convert Date to datetime and sort
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    
    # Convert numeric columns
    for col in ['Gross P&L', 'Charges', 'Net P&L']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sort by date (descending - most recent first)
    df = df.sort_values('Date', ascending=False)
    
    # Calculate cumulative metrics
    df['Cumulative Gross P&L'] = df['Gross P&L'].cumsum()
    df['Cumulative Net P&L'] = df['Net P&L'].cumsum()
    
    # Format date for display
    df['Date_Display'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    # Add region column
    df['Region'] = region
    
    return df


def create_daily_pnl_dashboard(daily_pnl_df, region="INDIA"):
    """Create Daily PnL dashboard with beautiful chart similar to intraday plot"""
    if daily_pnl_df.empty:
        st.info(f"📭 No Daily PnL data available for {region}")
        return
    
    # Get currency formatter
    format_currency_func = format_inr if region == "INDIA" else lambda x: format_currency(x, "$")
    currency_symbol = "₹" if region == "INDIA" else "$"
    
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <h2 style="color: #1f77b4;">📅 {region} Daily P&L Dashboard</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # ===================================================================
    # 📊 KEY METRICS
    # ===================================================================
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_gross = daily_pnl_df['Gross PnL'].sum()
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.85rem; font-weight: 600; color: #2ca02c; margin-bottom: 0.2rem;">Total Gross P&L</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #2ca02c;">{format_currency_func(total_gross)}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col2:
        total_charges = daily_pnl_df['Charges'].sum()
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.85rem; font-weight: 600; color: #d62728; margin-bottom: 0.2rem;">Total Charges</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #d62728;">{format_currency_func(total_charges)}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col3:
        total_net = daily_pnl_df['Net PnL'].sum()
        # Determine color for total net P&L
        net_color = "#2ca02c" if total_net >= 0 else "#d62728"
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.85rem; font-weight: 600; color: {net_color}; margin-bottom: 0.2rem;">Total Net P&L</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: {net_color};">{format_currency_func(total_net)}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col4:
        # Get current capital (latest date)
        if not daily_pnl_df.empty and 'Capital' in daily_pnl_df.columns:
            current_capital = daily_pnl_df['Capital'].iloc[-1]
            initial_capital = daily_pnl_df['Capital'].iloc[0]
            capital_growth = current_capital - initial_capital
            capital_color = "#2ca02c" if capital_growth >= 0 else "#d62728"
            
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <div style="font-size: 0.85rem; font-weight: 600; color: #ff7f0e; margin-bottom: 0.2rem;">Current Capital</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #ff7f0e;">{format_currency_func(current_capital)}</div>
                    <div style="font-size: 0.75rem; color: {capital_color};">
                        {format_currency_func(capital_growth)} ({capital_growth/initial_capital*100:+.1f}%)
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <div style="font-size: 0.85rem; font-weight: 600; color: #ff7f0e; margin-bottom: 0.2rem;">Current Capital</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #ff7f0e;">N/A</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
    
    st.divider()
    
    # ===================================================================
    # 📈 BEAUTIFUL DUAL-AXIS CHART: Net PnL (colored by value) & Capital
    # ===================================================================
    st.subheader("💰 Daily Performance")
    
    # Sort by Date (ascending for proper chart)
    daily_pnl_df_sorted = daily_pnl_df.sort_values('Date')
    
    if len(daily_pnl_df_sorted) > 0:
        # Find first occurrence of highest and lowest Net PnL values
        highest_value = daily_pnl_df_sorted['Net PnL'].max()
        lowest_value = daily_pnl_df_sorted['Net PnL'].min()
        
        # Get first occurrence of highest and lowest
        highest_row = daily_pnl_df_sorted[daily_pnl_df_sorted['Net PnL'] == highest_value].iloc[0]
        lowest_row = daily_pnl_df_sorted[daily_pnl_df_sorted['Net PnL'] == lowest_value].iloc[0]
        
        # Create the chart
        fig = go.Figure()
        
        # Create a single line for Net PnL that changes color based on value
        segments = []
        current_segment = {'x': [], 'y': [], 'color': None}
        
        for i in range(len(daily_pnl_df_sorted)):
            current_val = daily_pnl_df_sorted['Net PnL'].iloc[i]
            current_date = daily_pnl_df_sorted['Date'].iloc[i]
            current_color = '#10B981' if current_val >= 0 else '#EF4444'  # Green/Red
            
            if not current_segment['x']:
                # First point
                current_segment['x'].append(current_date)
                current_segment['y'].append(current_val)
                current_segment['color'] = current_color
            elif current_segment['color'] == current_color:
                # Same color, continue segment
                current_segment['x'].append(current_date)
                current_segment['y'].append(current_val)
            else:
                # Color changed (crossed zero), handle the transition
                prev_val = daily_pnl_df_sorted['Net PnL'].iloc[i-1]
                prev_date = daily_pnl_df_sorted['Date'].iloc[i-1]
                
                # Calculate the exact zero crossing point
                # For dates, we need to handle differently than time
                days_diff = (current_date - prev_date).days
                if days_diff > 0:
                    # Linear interpolation for zero crossing
                    zero_fraction = -prev_val / (current_val - prev_val) if (current_val - prev_val) != 0 else 0
                    zero_date = prev_date + pd.Timedelta(days=days_diff * zero_fraction)
                    
                    # Add point at zero to complete current segment
                    current_segment['x'].append(zero_date)
                    current_segment['y'].append(0)
                    segments.append(current_segment.copy())
                    
                    # Start new segment from zero point
                    current_segment = {
                        'x': [zero_date, current_date],
                        'y': [0, current_val],
                        'color': current_color
                    }
                else:
                    # Same date, just add point
                    current_segment['x'].append(current_date)
                    current_segment['y'].append(current_val)
        
        # Add the last segment
        if current_segment['x']:
            segments.append(current_segment)
        
        # Add all segments as separate traces for Net PnL
        for segment in segments:
            fig.add_trace(go.Scatter(
                x=segment['x'],
                y=segment['y'],
                mode='lines',
                line=dict(
                    shape='spline',
                    smoothing=1.0,
                    width=3,
                    color=segment['color']
                ),
                showlegend=False,
                hoverinfo='skip',
                yaxis='y'
            ))
        
        # Add invisible trace for hover information for Net PnL
        fig.add_trace(go.Scatter(
            x=daily_pnl_df_sorted['Date'],
            y=daily_pnl_df_sorted['Net PnL'],
            mode='lines',
            line=dict(width=0),  # Invisible
            hovertemplate=f'<b>%{{x|%Y-%m-%d}}</b><br>Net P&L: {currency_symbol}%{{y:,.2f}}<extra></extra>',
            showlegend=False,
            name='Net P&L',
            yaxis='y'
        ))
        
        # Add Capital line on secondary Y-axis (right side)
        if 'Capital' in daily_pnl_df_sorted.columns and not daily_pnl_df_sorted['Capital'].isnull().all():
            fig.add_trace(go.Scatter(
                x=daily_pnl_df_sorted['Date'],
                y=daily_pnl_df_sorted['Capital'],
                name='Capital',
                mode='lines',
                line=dict(color='#FFA500', width=3, dash='dash'),  # Orange dashed
                marker=dict(size=8, color='#FFA500'),
                hovertemplate=f'<b>%{{x|%Y-%m-%d}}</b><br>Capital: {currency_symbol}%{{y:,.2f}}<extra></extra>',
                yaxis='y2'
            ))
        
        # Add zero line
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="#94A3B8",
            line_width=1,
            opacity=0.3,
            yref="y"
        )
        
        # Add area fill with appropriate color for Net PnL
        x_full = daily_pnl_df_sorted['Date'].tolist()
        y_full = daily_pnl_df_sorted['Net PnL'].tolist()
        
        # Fill area for positive values
        fig.add_trace(go.Scatter(
            x=x_full,
            y=y_full,
            mode='none',
            fill='tozeroy',
            fillcolor='rgba(16, 185, 129, 0.1)',  # Light green for positive
            showlegend=False,
            hoverinfo='skip',
            yaxis='y'
        ))
        
        # Also fill negative area separately (overlay)
        fig.add_trace(go.Scatter(
            x=x_full,
            y=[min(y, 0) for y in y_full],  # Only negative part
            mode='none',
            fill='tozeroy',
            fillcolor='rgba(239, 68, 68, 0.1)',  # Light red for negative
            showlegend=False,
            hoverinfo='skip',
            yaxis='y'
        ))
        
        # ===================================================================
        # 🎯 ADD HIGHEST AND LOWEST POINT MARKERS FOR NET P&L
        # ===================================================================
        
        # Mark highest point (first occurrence)
        fig.add_trace(go.Scatter(
            x=[highest_row['Date']],
            y=[highest_value],
            mode='markers+text',
            marker=dict(
                size=12,
                color='#10B981',
                symbol='triangle-up',
                line=dict(width=2, color='white')
            ),
            text=[f"  High: {currency_symbol}{highest_value:,.0f}"],
            textposition="top center",
            textfont=dict(size=11, color='#10B981', family='Arial'),
            hovertemplate=f'<b>Highest Net P&L: {currency_symbol}{highest_value:,.2f}</b><br>Date: %{{x|%Y-%m-%d}}<extra></extra>',
            showlegend=False,
            name='Highest',
            yaxis='y'
        ))
        
        # Mark lowest point (first occurrence)
        fig.add_trace(go.Scatter(
            x=[lowest_row['Date']],
            y=[lowest_value],
            mode='markers+text',
            marker=dict(
                size=12,
                color='#EF4444',
                symbol='triangle-down',
                line=dict(width=2, color='white')
            ),
            text=[f"  Low: {currency_symbol}{lowest_value:,.0f}"],
            textposition="bottom center",
            textfont=dict(size=11, color='#EF4444', family='Arial'),
            hovertemplate=f'<b>Lowest Net P&L: {currency_symbol}{lowest_value:,.2f}</b><br>Date: %{{x|%Y-%m-%d}}<extra></extra>',
            showlegend=False,
            name='Lowest',
            yaxis='y'
        ))
        
        # Clean layout with dual y-axes
        fig.update_layout(
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, system-ui, sans-serif", size=12),
            hovermode='x unified',
            margin=dict(l=60, r=60, t=40, b=60),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(
                title="Date",
                tickformat='%b %d',
                showgrid=False,
                tickfont=dict(size=10, color='#64748B'),
                linecolor='#E2E8F0',
                showline=True
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
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Show latest values
        if len(daily_pnl_df_sorted) > 0:
            latest_data = daily_pnl_df_sorted.iloc[-1]
            latest_net_pnl = latest_data['Net PnL']
            latest_capital = latest_data.get('Capital', 0)
            latest_date = latest_data['Date'].strftime('%Y-%m-%d')
            pnl_color = "#10B981" if latest_net_pnl >= 0 else "#EF4444"
            capital_color = "#10B981" if latest_capital >= 0 else "#EF4444"
            
            st.markdown(
                f"""
                <div style="text-align: center; margin-top: 10px; padding: 10px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid {pnl_color};">
                    <span style="font-size: 1.1rem; font-weight: 600; color: {pnl_color};">
                        📊 Latest Daily Net P&L: {format_currency_func(latest_net_pnl)} on {latest_date}
                    </span>
                    <br>
                    <span style="font-size: 1rem; font-weight: 600; color: #FFA500;">
                        💰 Current Capital: {format_currency_func(latest_capital)}
                    </span>
                    <br>
                    <span style="font-size: 0.9rem; color: #64748B;">
                        Best Day: {format_currency_func(highest_value)} • Worst Day: {format_currency_func(lowest_value)}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    st.divider()
    
    # ===================================================================
    # 📋 DAILY P&L TABLE
    # ===================================================================
    st.subheader("📋 Daily P&L Details")
    
    # Prepare display dataframe (show all days, sorted by date descending)
    display_df = daily_pnl_df.sort_values('Date', ascending=False).copy()
    
    # Create a nice table with color coding
    html_table = """
    <div style="overflow-x: auto;">
    <table style="width: 100%; border-collapse: collapse; margin: 10px 0; font-family: Arial, sans-serif;">
        <thead>
            <tr style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                <th style="padding: 12px; text-align: left; border-bottom: 2px solid #ddd;">Date</th>
                <th style="padding: 12px; text-align: right; border-bottom: 2px solid #ddd;">Gross P&L</th>
                <th style="padding: 12px; text-align: right; border-bottom: 2px solid #ddd;">Charges</th>
                <th style="padding: 12px; text-align: right; border-bottom: 2px solid #ddd;">Net P&L</th>
                <th style="padding: 12px; text-align: right; border-bottom: 2px solid #ddd;">Capital</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for _, row in display_df.iterrows():
        html_table += '<tr style="border-bottom: 1px solid #e0e0e0;">'
        
        # Date
        html_table += f'<td style="padding: 10px; font-weight: 600;">{row["Date_Display"]}</td>'
        
        # Gross P&L (color based on value)
        gross_pnl = row["Gross PnL"]
        gross_color = "#2ca02c" if gross_pnl >= 0 else "#d62728"
        html_table += f'<td style="padding: 10px; text-align: right; color: {gross_color}; font-weight: 600;">{format_currency_func(gross_pnl)}</td>'
        
        # Charges (always red)
        html_table += f'<td style="padding: 10px; text-align: right; color: #d62728; font-weight: 600;">{format_currency_func(row["Charges"])}</td>'
        
        # Net P&L (color based on value)
        net_pnl = row["Net PnL"]
        net_color = "#2ca02c" if net_pnl >= 0 else "#d62728"
        html_table += f'<td style="padding: 10px; text-align: right; color: {net_color}; font-weight: 700;">{format_currency_func(net_pnl)}</td>'
        
        # Capital (orange)
        capital = row.get('Capital', 0)
        html_table += f'<td style="padding: 10px; text-align: right; color: #FFA500; font-weight: 600;">{format_currency_func(capital)}</td>'
        
        html_table += '</tr>'
    
    html_table += """
        </tbody>
    </table>
    </div>
    """
    
    st.markdown(html_table, unsafe_allow_html=True)
    
    # Show download option
    csv = daily_pnl_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Daily P&L Data",
        data=csv,
        file_name=f"{region.lower()}_daily_pnl.csv",
        mime="text/csv",
        type="primary"
    )
    


# ===================================================================
# 📥 Load & Clean Data — WITH AUTOMATIC REFRESH
# ===================================================================

# Load data for all sheets
df_india_raw = load_sheet_data(sheet_gid="649765105")  # INDIA sheet
df_india_live_pnl_raw = load_sheet_data(sheet_gid="1065660372")  # INDIA LIVE PnL sheet
df_india_daily_pnl_raw = load_sheet_data(sheet_gid="795838620")  # INDIA Daily PnL sheet

# Load Global sheets
df_global_raw = load_sheet_data(sheet_gid="94252270")  # IB_GLOBAL sheet
df_global_live_pnl_raw = load_sheet_data(sheet_gid="1297846329")  # IB_GLOBAL_LIVE_PnL sheet
df_global_daily_pnl_raw = load_sheet_data(sheet_gid="1869867842")  # GLOBAL Daily PnL sheet

# Process data
india_data = process_india_data(df_india_raw)
india_live_pnl_data = process_live_pnl_data(df_india_live_pnl_raw)
india_daily_pnl_data = process_daily_pnl_data(df_india_daily_pnl_raw, region="INDIA")

# Process Global data
if df_global_raw.empty:
    global_data = {'open_positions': pd.DataFrame(), 'closed_positions': pd.DataFrame(), 'summary': {}}
else:
    global_data = process_india_data(df_global_raw)

if df_global_live_pnl_raw.empty:
    global_live_pnl_data = pd.DataFrame()
else:
    global_live_pnl_data = process_live_pnl_data(df_global_live_pnl_raw)

global_daily_pnl_data = process_daily_pnl_data(df_global_daily_pnl_raw, region="GLOBAL")

# ===================================================================
# 🎨 CSS for Bigger Tabs
# ===================================================================
st.markdown("""
<style>
    /* Make tabs larger and more prominent */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        padding-top: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        font-size: 18px;
        font-weight: 600;
        padding: 10px 24px;
        border-radius: 8px 8px 0 0;
        background-color: #f0f2f6;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        border-bottom: 3px solid #FF4B4B;
    }
    
    /* Remove default Streamlit tab styling */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1px solid #e0e0e0;
    }
    
    /* Make the tab content area cleaner */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1rem;
    }
    
    /* Remove extra spacing in main container */
    .main .block-container {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ===================================================================
# 📊 Create Tabs - Simplified with just 4 tabs
# ===================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🌍 **GLOBAL**", 
    "🇮🇳 **INDIA**",
    "📊 **GLOBAL DAILY**",
    "📊 **INDIA DAILY**"
])

with tab1:
    # Refresh button at top-right of Global dashboard
    gcol1, gcol2 = st.columns([5, 1])
    with gcol1:
        st.write("")  # Empty
    with gcol2:
        if st.button("🔄 Refresh Data", type="secondary", key="refresh_global"):
            st.cache_data.clear()
            st.rerun()
    
    # Use the new dashboard function for Global
    create_dashboard(global_data, global_live_pnl_data, region="GLOBAL")

with tab2:
    # Refresh button at top-right of India dashboard
    icol1, icol2 = st.columns([5, 1])
    with icol1:
        st.write("")  # Empty
    with icol2:
        if st.button("🔄 Refresh Data", type="secondary", key="refresh_india"):
            st.cache_data.clear()
            st.rerun()
    
    # Use the new dashboard function for India
    create_dashboard(india_data, india_live_pnl_data, region="INDIA")

with tab3:
    # Refresh button for Global Daily PnL
    gdcol1, gdcol2 = st.columns([5, 1])
    with gdcol1:
        st.write("")  # Empty
    with gdcol2:
        if st.button("🔄 Refresh Data", type="secondary", key="refresh_global_daily"):
            st.cache_data.clear()
            st.rerun()
    
    # Create Daily PnL dashboard for Global
    create_daily_pnl_dashboard(global_daily_pnl_data, region="GLOBAL")

with tab4:
    # Refresh button for India Daily PnL
    idcol1, idcol2 = st.columns([5, 1])
    with idcol1:
        st.write("")  # Empty
    with idcol2:
        if st.button("🔄 Refresh Data", type="secondary", key="refresh_india_daily"):
            st.cache_data.clear()
            st.rerun()
    
    # Create Daily PnL dashboard for India
    create_daily_pnl_dashboard(india_daily_pnl_data, region="INDIA")
