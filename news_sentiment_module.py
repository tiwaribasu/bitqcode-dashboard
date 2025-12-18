# news_sentiment_module.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import re
import plotly.graph_objects as go
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

class NewsSentimentAnalyzer:
    def __init__(self, google_sheet_url=None):
        """
        Initialize the News Sentiment Analyzer
        
        Args:
            google_sheet_url: URL of the Google Sheet
        """
        self.google_sheet_url = google_sheet_url
        self.df = None
        
    def clean_text(self, text):
        """Remove emojis, special characters, and clean text for NLP"""
        if pd.isna(text):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove Twitter handles
        text = re.sub(r'@\w+', '', text)
        
        # Remove emojis
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002700-\U000027BF"  # Dingbats
            u"\U000024C2-\U0001F251" 
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        
        # Remove RT (retweet) mentions
        text = re.sub(r'\bRT\b', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def parse_datetime(self, datetime_str):
        """Parse datetime string to datetime object in ET timezone"""
        try:
            # Clean the datetime string
            datetime_str = str(datetime_str).strip()
            
            # Handle different formats
            try:
                # Format: "December 18, 2025 at 05:15PM"
                dt = datetime.strptime(datetime_str, "%B %d, %Y at %I:%M%p")
            except ValueError:
                try:
                    # Try without "at"
                    dt = datetime.strptime(datetime_str, "%B %d, %Y %I:%M%p")
                except ValueError:
                    # Try another format
                    dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
            
            # Convert to ET (UTC-5)
            et_offset = timedelta(hours=-5)
            dt_et = dt + et_offset
            
            return dt_et
            
        except Exception as e:
            st.warning(f"Could not parse datetime: {datetime_str} - Error: {str(e)}")
            return datetime.now(timezone.utc)
    
    def load_news_data(self):
        """Load news data from Google Sheet"""
        try:
            if self.google_sheet_url:
                # Convert Google Sheet URL to CSV export URL
                if '/edit#' in self.google_sheet_url:
                    # Extract gid if present
                    if 'gid=' in self.google_sheet_url:
                        gid = self.google_sheet_url.split('gid=')[1]
                        csv_url = self.google_sheet_url.replace('/edit#gid=', f'/export?format=csv&gid=')
                    else:
                        csv_url = self.google_sheet_url.replace('/edit#', '/export?format=csv&gid=0')
                else:
                    csv_url = self.google_sheet_url
                
                # Make sure it's a CSV export URL
                if 'export?format=csv' not in csv_url:
                    if 'gid=' in csv_url:
                        csv_url = csv_url.replace('/edit?', '/export?format=csv&')
                    else:
                        csv_url = f"{csv_url}/export?format=csv"
                
                self.df = pd.read_csv(csv_url)
                
                if self.df is not None and not self.df.empty:
                    # Clean column names
                    self.df.columns = [col.strip() for col in self.df.columns]
                    
                    # Clean news text
                    if 'News' in self.df.columns:
                        self.df['Cleaned_News'] = self.df['News'].apply(self.clean_text)
                    
                    # Parse datetime
                    if 'DateTime' in self.df.columns:
                        self.df['DateTime_ET'] = self.df['DateTime'].apply(self.parse_datetime)
                        
                        # Filter for today's news only
                        today = datetime.now(timezone.utc).date()
                        self.df['Date'] = pd.to_datetime(self.df['DateTime_ET']).dt.date
                        self.df = self.df[self.df['Date'] == today]
                        
                        # Sort by datetime (newest first)
                        self.df = self.df.sort_values('DateTime_ET', ascending=False)
                    
                    return True
            return False
            
        except Exception as e:
            st.error(f"Error loading news data: {str(e)}")
            return False
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using TextBlob"""
        if not text or pd.isna(text):
            return {'score': 50, 'sentiment': 'Neutral', 'color': 'yellow', 'polarity': 0}
        
        analysis = TextBlob(str(text))
        # Get polarity score between -1 and 1
        polarity = analysis.sentiment.polarity
        
        # Convert to score between 0 and 100
        score = (polarity + 1) * 50
        
        # Categorize sentiment
        if score > 60:
            sentiment = "Positive"
            color = "#10B981"  # Green
        elif score < 40:
            sentiment = "Negative"
            color = "#EF4444"  # Red
        else:
            sentiment = "Neutral"
            color = "#F59E0B"  # Yellow/Amber
        
        return {
            'score': min(max(score, 0), 100),  # Ensure between 0-100
            'sentiment': sentiment,
            'color': color,
            'polarity': polarity
        }
    
    def create_speedometer(self, sentiment_score, sentiment_label, color):
        """Create a speedometer/gauge chart for sentiment"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sentiment_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Market Sentiment: {sentiment_label}", 'font': {'size': 20, 'color': '#1E293B'}},
            number={'font': {'size': 48, 'color': color}, 'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#1E293B", 'tickfont': {'size': 12}},
                'bar': {'color': color, 'thickness': 0.8},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#CBD5E1",
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(239, 68, 68, 0.2)'},
                    {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.2)'},
                    {'range': [70, 100], 'color': 'rgba(16, 185, 129, 0.2)'}
                ],
                'threshold': {
                    'line': {'color': color, 'width': 4},
                    'thickness': 0.8,
                    'value': sentiment_score
                }
            }
        ))
        
        fig.update_layout(
            height=400,
            margin=dict(l=30, r=30, t=80, b=30),
            font={'family': "Inter, system-ui, sans-serif"},
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        return fig
    
    def calculate_overall_sentiment(self):
        """Calculate overall sentiment from all news"""
        if self.df is None or self.df.empty or 'Cleaned_News' not in self.df.columns:
            return {'score': 50, 'sentiment': 'Neutral', 'color': '#F59E0B'}
        
        # Use all cleaned news for overall sentiment
        all_news = " ".join(self.df['Cleaned_News'].dropna().tolist())
        if not all_news.strip():
            return {'score': 50, 'sentiment': 'Neutral', 'color': '#F59E0B'}
        
        return self.analyze_sentiment(all_news)
    
    def display_dashboard(self):
        """Display the complete news sentiment dashboard"""
        st.title("📰 NEWS & SENTIMENT DASHBOARD")
        
        # Load data
        with st.spinner("📥 Loading news data..."):
            if self.load_news_data():
                if self.df is not None and not self.df.empty:
                    st.success(f"✓ Loaded {len(self.df)} news items for today")
                else:
                    st.info("📭 No news available for today.")
                    return
            else:
                st.error("❌ Failed to load news data")
                return
        
        # Calculate overall sentiment
        overall_sentiment = self.calculate_overall_sentiment()
        
        # Create two columns layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Speedometer section
            st.subheader("📊 Market Sentiment Indicator")
            
            # Display speedometer
            fig = self.create_speedometer(
                overall_sentiment['score'],
                overall_sentiment['sentiment'],
                overall_sentiment['color']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment statistics
            st.markdown("---")
            st.subheader("📈 Sentiment Breakdown")
            
            if not self.df.empty:
                # Calculate sentiment distribution
                sentiments = []
                for news in self.df['Cleaned_News']:
                    if pd.notna(news) and str(news).strip():
                        sentiment = self.analyze_sentiment(news)
                        sentiments.append(sentiment['sentiment'])
                
                if sentiments:
                    sentiment_counts = pd.Series(sentiments).value_counts()
                    
                    # Display sentiment counts with colors
                    for sentiment_type in ['Positive', 'Neutral', 'Negative']:
                        count = sentiment_counts.get(sentiment_type, 0)
                        if sentiment_type == "Positive":
                            icon = "🟢"
                            color = "#10B981"
                        elif sentiment_type == "Negative":
                            icon = "🔴"
                            color = "#EF4444"
                        else:
                            icon = "🟡"
                            color = "#F59E0B"
                        
                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span style="font-size: 16px; color: #1E293B;">
                                {icon} <strong>{sentiment_type}</strong>
                            </span>
                            <span style="font-size: 16px; font-weight: bold; color: {color};">
                                {count}
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Latest update time
                st.markdown("---")
                if not self.df.empty and 'DateTime_ET' in self.df.columns:
                    latest_news_time = self.df['DateTime_ET'].iloc[0]
                    st.caption(f"""
                    <div style="text-align: center; color: #64748B; font-size: 12px;">
                        📅 Last update: {latest_news_time.strftime('%Y-%m-%d %H:%M:%S ET')}
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            # News feed section
            st.subheader("📰 Live News Feed")
            
            # Create Bloomberg-like terminal display
            if self.df is not None and not self.df.empty:
                # Container for news feed
                news_container = st.container()
                
                with news_container:
                    # Create HTML for terminal-like display
                    html_content = """
                    <div style="
                        font-family: 'Courier New', Monaco, monospace;
                        background-color: #000000;
                        color: #00FF00;
                        padding: 20px;
                        border-radius: 8px;
                        border: 1px solid #00FF00;
                        max-height: 600px;
                        overflow-y: auto;
                        line-height: 1.6;
                    ">
                    """
                    
                    for idx, row in self.df.iterrows():
                        if idx >= 50:  # Limit to 50 most recent news items
                            break
                            
                        timestamp = row['DateTime_ET'].strftime("%H:%M:%S")
                        news_text = row['Cleaned_News']
                        
                        if not news_text or str(news_text).strip() == "":
                            continue
                        
                        # Analyze sentiment for this specific news
                        news_sentiment = self.analyze_sentiment(news_text)
                        
                        # Color code based on sentiment
                        if news_sentiment['sentiment'] == "Positive":
                            sentiment_color = "#00FF00"  # Green
                            sentiment_indicator = "▲"
                        elif news_sentiment['sentiment'] == "Negative":
                            sentiment_color = "#FF0000"  # Red
                            sentiment_indicator = "▼"
                        else:
                            sentiment_color = "#FFFF00"  # Yellow
                            sentiment_indicator = "●"
                        
                        html_content += f"""
                        <div style="margin-bottom: 15px; border-left: 3px solid {sentiment_color}; padding-left: 10px;">
                            <div style="color: #888888; font-size: 12px; margin-bottom: 4px;">
                                [{timestamp}] <span style="color: {sentiment_color}; font-weight: bold;">{sentiment_indicator}</span>
                            </div>
                            <div style="color: #FFFFFF; font-size: 14px; line-height: 1.4;">
                                {news_text}
                            </div>
                        </div>
                        """
                    
                    html_content += "</div>"
                    
                    st.markdown(html_content, unsafe_allow_html=True)
                    
                    # News statistics
                    st.markdown("---")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("📊 Total News", len(self.df))
                    with col_b:
                        if not self.df.empty:
                            latest_time = self.df['DateTime_ET'].iloc[0].strftime("%H:%M ET")
                            st.metric("🕒 Latest", latest_time)
                    with col_c:
                        st.metric("📈 Sentiment", f"{overall_sentiment['score']:.1f}%")
            else:
                st.info("No news items to display.")
