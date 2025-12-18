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
        
        # Financial keywords for sentiment boosting
        self.positive_keywords = [
            'beat', 'surge', 'jump', 'rise', 'gain', 'rally', 'bull', 'positive',
            'growth', 'profit', 'increase', 'higher', 'record', 'win', 'success',
            'strong', 'optimistic', 'boom', 'breakthrough', 'dividend', 'buyback',
            'upgrade', 'outperform', 'bullish', 'recovery', 'soar', 'lifeline',
            'cut rates', 'break', 'breakout', 'outperform', 'beat estimates'
        ]
        
        self.negative_keywords = [
            'cut', 'plunge', 'drop', 'fall', 'loss', 'crash', 'bear', 'negative',
            'decline', 'decrease', 'lower', 'miss', 'fail', 'weak', 'pessimistic',
            'slump', 'downgrade', 'underperform', 'bearish', 'recession', 'warn',
            'risk', 'volatility', 'uncertainty', 'selloff', 'downturn', 'bankrupt',
            'soaring', 'beating', 'crisis', 'tumble', 'plummet', 'collapse', 'slump'
        ]
    
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
                    try:
                        # Try another format with seconds
                        dt = datetime.strptime(datetime_str, "%B %d, %Y at %I:%M:%S%p")
                    except ValueError:
                        # Try simple date format
                        dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
            
            # Convert to ET (UTC-5)
            et_offset = timedelta(hours=-5)
            dt_et = dt + et_offset
            
            return dt_et
            
        except Exception as e:
            st.warning(f"Could not parse datetime: {datetime_str}")
            return datetime.now(timezone.utc)
    
    def load_news_data(self):
        """Load news data from Google Sheet"""
        try:
            if self.google_sheet_url:
                # For public Google Sheets
                if '/edit#' in self.google_sheet_url:
                    # Extract sheet ID
                    if 'gid=' in self.google_sheet_url:
                        sheet_id = self.google_sheet_url.split('gid=')[1]
                        csv_url = self.google_sheet_url.replace('/edit#gid=', f'/export?format=csv&gid={sheet_id}')
                    else:
                        csv_url = self.google_sheet_url.replace('/edit#', '/export?format=csv&gid=0')
                else:
                    # Assume it's already a CSV export URL
                    csv_url = self.google_sheet_url
                
                # Make sure it's a CSV export URL
                if 'export?format=csv' not in csv_url:
                    if '/edit?' in csv_url:
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
                        # Remove empty news
                        self.df = self.df[self.df['Cleaned_News'].str.strip() != '']
                    
                    # Parse datetime
                    if 'DateTime' in self.df.columns:
                        self.df['DateTime_ET'] = self.df['DateTime'].apply(self.parse_datetime)
                        
                        # Filter for today's news only
                        today = datetime.now(timezone.utc).date()
                        self.df['Date'] = pd.to_datetime(self.df['DateTime_ET']).dt.date
                        
                        # Check if we have today's news
                        if today in self.df['Date'].unique():
                            self.df = self.df[self.df['Date'] == today]
                        else:
                            # Show most recent date's news
                            most_recent_date = self.df['Date'].max()
                            self.df = self.df[self.df['Date'] == most_recent_date]
                        
                        # Sort by datetime (newest first)
                        self.df = self.df.sort_values('DateTime_ET', ascending=False)
                    
                    return True
            return False
            
        except Exception as e:
            st.error(f"Error loading news data: {str(e)}")
            return False
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using TextBlob with financial keyword boosting"""
        if not text or pd.isna(text):
            return {'score': 50, 'sentiment': 'Neutral', 'color': '#FFFF00', 'indicator': '●'}
        
        text_lower = str(text).lower()
        
        # Base sentiment from TextBlob
        analysis = TextBlob(str(text))
        polarity = analysis.sentiment.polarity
        
        # Adjust based on financial keywords
        positive_count = sum(1 for keyword in self.positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in self.negative_keywords if keyword in text_lower)
        
        # Boost polarity based on keyword matches
        keyword_boost = 0
        if positive_count > negative_count:
            keyword_boost = 0.25  # Boost for positive keywords
        elif negative_count > positive_count:
            keyword_boost = -0.25  # Reduce for negative keywords
        
        # Apply keyword boost
        polarity = max(-1.0, min(1.0, polarity + keyword_boost))
        
        # Convert to score between 0 and 100
        score = (polarity + 1) * 50
        
        # Adjust thresholds for better distribution
        if score > 55:  # More sensitive positive threshold
            sentiment = "Positive"
            color = "#00FF00"  # Bright green for terminal
            indicator = "▲"
        elif score < 45:  # More sensitive negative threshold
            sentiment = "Negative"
            color = "#FF0000"  # Bright red for terminal
            indicator = "▼"
        else:
            sentiment = "Neutral"
            color = "#FFFF00"  # Yellow for terminal
            indicator = "●"
        
        return {
            'score': min(max(score, 0), 100),
            'sentiment': sentiment,
            'color': color,
            'indicator': indicator,
            'polarity': polarity,
            'positive_keywords': positive_count,
            'negative_keywords': negative_count
        }
    
    def create_speedometer(self, sentiment_score, sentiment_label, color):
        """Create a speedometer/gauge chart for sentiment"""
        # Define color based on sentiment
        if sentiment_label == "Positive":
            gauge_color = "#10B981"
        elif sentiment_label == "Negative":
            gauge_color = "#EF4444"
        else:
            gauge_color = "#F59E0B"
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sentiment_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Market Sentiment", 'font': {'size': 24, 'color': '#1E293B'}},
            number={'font': {'size': 48, 'color': gauge_color}, 'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#1E293B", 
                        'tickfont': {'size': 12, 'color': '#64748B'}, 'tickmode': 'array',
                        'tickvals': [0, 25, 50, 75, 100], 'ticktext': ['Very Bearish', 'Bearish', 'Neutral', 'Bullish', 'Very Bullish']},
                'bar': {'color': gauge_color, 'thickness': 0.8},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#CBD5E1",
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(239, 68, 68, 0.1)'},
                    {'range': [30, 45], 'color': 'rgba(245, 158, 11, 0.1)'},
                    {'range': [45, 55], 'color': 'rgba(251, 191, 36, 0.1)'},
                    {'range': [55, 70], 'color': 'rgba(245, 158, 11, 0.1)'},
                    {'range': [70, 100], 'color': 'rgba(16, 185, 129, 0.1)'}
                ],
                'threshold': {
                    'line': {'color': gauge_color, 'width': 4},
                    'thickness': 0.8,
                    'value': sentiment_score
                }
            }
        ))
        
        # Add sentiment label
        fig.add_annotation(
            x=0.5,
            y=0.2,
            text=f"<b>{sentiment_label}</b>",
            showarrow=False,
            font=dict(size=20, color=gauge_color),
            xref="paper",
            yref="paper"
        )
        
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
            return {'score': 50, 'sentiment': 'Neutral', 'color': '#F59E0B', 'indicator': '●'}
        
        # Calculate weighted sentiment (recent news more important)
        if len(self.df) > 0:
            sentiments = []
            weights = []
            
            for i, (idx, row) in enumerate(self.df.iterrows()):
                if pd.notna(row['Cleaned_News']) and str(row['Cleaned_News']).strip():
                    sentiment = self.analyze_sentiment(row['Cleaned_News'])
                    # Weight: recent news gets higher weight
                    weight = 1.0 / (i + 1)  # Linear decay
                    sentiments.append(sentiment['score'])
                    weights.append(weight)
            
            if sentiments:
                weighted_avg = np.average(sentiments, weights=weights)
                
                if weighted_avg > 55:
                    sentiment_label = "Positive"
                    color = "#10B981"
                elif weighted_avg < 45:
                    sentiment_label = "Negative"
                    color = "#EF4444"
                else:
                    sentiment_label = "Neutral"
                    color = "#F59E0B"
                
                return {
                    'score': weighted_avg,
                    'sentiment': sentiment_label,
                    'color': color,
                    'indicator': '▲' if sentiment_label == "Positive" else '▼' if sentiment_label == "Negative" else '●'
                }
        
        return {'score': 50, 'sentiment': 'Neutral', 'color': '#F59E0B', 'indicator': '●'}
    
    def display_dashboard(self):
        """Display the complete news sentiment dashboard"""
        # Load data
        with st.spinner("📥 Loading news data..."):
            if self.load_news_data():
                if self.df is None and self.df.empty:
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
                    sentiment_data = []
                    for sentiment_type, bg_color, icon in [
                        ('Positive', 'rgba(16, 185, 129, 0.1)', '🟢'),
                        ('Neutral', 'rgba(245, 158, 11, 0.1)', '🟡'),
                        ('Negative', 'rgba(239, 68, 68, 0.1)', '🔴')
                    ]:
                        count = sentiment_counts.get(sentiment_type, 0)
                        percentage = (count / len(sentiments)) * 100 if len(sentiments) > 0 else 0
                        
                        text_color = "#10B981" if sentiment_type == "Positive" else "#EF4444" if sentiment_type == "Negative" else "#F59E0B"
                        
                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; padding: 8px; background-color: {bg_color}; border-radius: 6px;">
                            <span style="font-size: 16px; color: #1E293B;">
                                {icon} <strong>{sentiment_type}</strong>
                            </span>
                            <div style="text-align: right;">
                                <div style="font-size: 18px; font-weight: bold; color: {text_color};">
                                    {count}
                                </div>
                                <div style="font-size: 12px; color: #64748B;">
                                    {percentage:.1f}%
                                </div>
                            </div>
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
                # Build the complete HTML
                html_content = ""
                
                # Add news items
                for idx, row in self.df.iterrows():
                    if idx >= 50:
                        break
                    
                    timestamp = row['DateTime_ET'].strftime("%H:%M:%S") if 'DateTime_ET' in row else "N/A"
                    news_text = row['Cleaned_News']
                    
                    if not news_text or str(news_text).strip() == "":
                        continue
                    
                    # Analyze sentiment
                    news_sentiment = self.analyze_sentiment(news_text)
                    
                    # Get color and indicator
                    sentiment_color = news_sentiment['color']
                    sentiment_indicator = news_sentiment['indicator']
                    
                    # Create HTML for this item
                    html_content += f"""
                    <div style="margin-bottom: 15px; border-left: 3px solid {sentiment_color}; padding-left: 10px; font-family: 'Courier New', monospace;">
                        <div style="color: #888888; font-size: 12px; margin-bottom: 4px;">
                            [{timestamp}] <span style="color: {sentiment_color}; font-weight: bold;">{sentiment_indicator}</span>
                        </div>
                        <div style="color: #FFFFFF; font-size: 14px; line-height: 1.4;">
                            {news_text}
                        </div>
                    </div>
                    """
                
                # Wrap in terminal container
                terminal_html = f"""
                <div style="background-color: #000000; color: #00FF00; padding: 20px; border-radius: 8px; border: 1px solid #00FF00; max-height: 600px; overflow-y: auto; font-family: 'Courier New', monospace;">
                    {html_content}
                </div>
                """
                
                # Display using st.write with HTML
                st.write(terminal_html, unsafe_allow_html=True)
                
                # News statistics
                st.markdown("---")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("📊 Total News", len(self.df))
                with col_b:
                    if not self.df.empty and 'DateTime_ET' in self.df.columns:
                        latest_time = self.df['DateTime_ET'].iloc[0].strftime("%H:%M ET")
                        st.metric("🕒 Latest", latest_time)
                with col_c:
                    st.metric("📈 Sentiment", f"{overall_sentiment['score']:.1f}%")
            else:
                st.info("No news items to display.")
