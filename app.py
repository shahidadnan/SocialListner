import streamlit as st
import pandas as pd
import json
import sys
import traceback
from typing import Dict, List, Optional
import io
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from utube_comment_replies_scrap import get_all_comments, save_to_csv
from urllib.parse import urlparse, parse_qs
import io
import csv

        

# Check for optional dependencies
PLOTLY_AVAILABLE = False
GROQ_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    pass

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    pass

# Configure Streamlit page
st.set_page_config(
    page_title="YouTube Comment Insight Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
class Theme(BaseModel):
    theme: str = Field(description="The identified theme or topic")
    frequency: int = Field(description="How often this theme appears")
    sentiment: str = Field(description="Overall sentiment: Positive, Negative, or Neutral")

class LovedElement(BaseModel):
    element: str = Field(description="The loved element or aspect")
    quotes: List[str] = Field(description="Sample quotes supporting this element")
    business_value: str = Field(description="Business value or insight")

class ConcernRisk(BaseModel):
    concern: str = Field(description="The concern or risk identified")
    quotes: List[str] = Field(description="Sample quotes highlighting this concern")
    mitigation: str = Field(description="Suggested mitigation strategy")

class BusinessRecommendation(BaseModel):
    action: str = Field(description="Recommended action to take")
    rationale: str = Field(description="Why this action is recommended")
    priority: str = Field(description="Priority level: High, Medium, or Low")

class ThemeAnalysis(BaseModel):
    top_themes: List[Theme] = Field(description="Top themes identified in comments")
    loved_elements: List[LovedElement] = Field(description="Elements that audience loves")
    concerns_risks: List[ConcernRisk] = Field(description="Concerns and risks identified")
    business_recommendations: List[BusinessRecommendation] = Field(description="Business recommendations")


# Define Pydantic models for structured output
class SentimentResult(BaseModel):
    comment_number: int = Field(description="The comment number")
    sentiment: str = Field(description="Sentiment classification: Positive, Negative, or Neutral")
    confidence: float = Field(description="Confidence score between 0 and 1")

class SentimentSummary(BaseModel):
    positive_count: int = Field(description="Number of positive comments")
    negative_count: int = Field(description="Number of negative comments")
    neutral_count: int = Field(description="Number of neutral comments")
    dominant_sentiment: str = Field(description="The most common sentiment")

class SentimentAnalysis(BaseModel):
    sentiments: List[SentimentResult] = Field(description="List of sentiment results for each comment")
    overall_summary: SentimentSummary = Field(description="Overall sentiment summary")


class CommentAnalyzer:
    def __init__(self, groq_api_key: str):
        if not GROQ_AVAILABLE:
            raise ImportError("Groq library not installed")
        self.groq_api_key = groq_api_key
        self.client = Groq(api_key=groq_api_key)
        
    def analyze_sentiment_batch(self, comments: List[str]) -> Optional[Dict]:
        """Analyze sentiment for a batch of comments"""
        my_list=comments
        my_list = my_list[:200] if len(my_list) > 200 else my_list
        comments_text = "\n".join([f"{i+1}. {comment[:200]}..." if len(comment) > 200 else f"{i+1}. {comment}" for i, comment in enumerate(my_list)])
        
        model= ChatGroq(model = 'llama-3.3-70b-versatile',max_tokens=2000, groq_api_key=self.groq_api_key)

        parser = JsonOutputParser(pydantic_object=SentimentAnalysis)

        prompt_template = ChatPromptTemplate.from_messages([
        ("human", """
        Analyze the sentiment of these YouTube comments and categorize them as Positive, Negative, or Neutral.
        
        Comments:
        {comments_text}
        
        {format_instructions}
        
        Be concise and accurate. Analyze each comment's sentiment with a confidence score between 0 and 1.
        """)
        ])
    
    # Create the chain
        chain = prompt_template | model | parser

        try:
        # Execute the chain
            result = chain.invoke({
            "comments_text": comments_text,
            "format_instructions": parser.get_format_instructions()
            })
        
            return result
        
        except Exception as e:
            st.error(f"Oops it's me AI I have please wait to complete the process: {str(e)}")
            return None
        

       
    def extract_themes_and_insights(self, comments: List[str], video_type: str = "general") -> Optional[Dict]:
        """Extract themes and business insights from comments using LangChain with Groq"""

        # Analyze top 30 comments for themes
        comments_sample = comments[:200]
        comments_text = "\n".join([
            f"- {comment[:150]}..." if len(comment) > 150 else f"- {comment}" 
            for comment in comments_sample
        ])

        # Initialize Groq LLM
        llm= ChatGroq(model = 'llama-3.3-70b-versatile',max_tokens=2000, groq_api_key=self.groq_api_key)

        # Set up the JSON output parser
        parser = JsonOutputParser(pydantic_object=ThemeAnalysis)

        # Define analysis focus based on video type
        if video_type == "music":
            analysis_focus = """
            Focus on music-specific insights:
            - Musical elements (beat, melody, vocals, production quality)
            - Visual elements (music video, aesthetics, styling)
            - Artist performance and appearance
            - Lyrics and messaging
            - Comparisons to other songs/artists
            """
        else:
            analysis_focus = """
            Focus on general content insights:
            - Content quality and production value
            - Messaging and communication effectiveness
            - Audience engagement and emotional response
            - Technical aspects and presentation
            """

        # Create the prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("human", """
            Analyze these YouTube comments for business insights and themes.

            {analysis_focus}

            Comments:
            {comments_text}

            {format_instructions}

            Be specific and actionable in your analysis. Identify concrete themes, provide relevant quotes, 
            and suggest practical business recommendations based on the feedback.
            """)
        ])

        # Create the chain
        chain = prompt_template | llm | parser

        try:
            # Execute the chain
            result = chain.invoke({
                "analysis_focus": analysis_focus,
                "comments_text": comments_text,
                "format_instructions": parser.get_format_instructions()
            })

            return result

        except Exception as e:
            st.error(f"Error in theme analysis: {str(e)}")
            return None

def create_sentiment_chart(sentiment_data: Dict):
    """Create a pie chart for sentiment distribution"""
    if not PLOTLY_AVAILABLE:
        # Fallback to simple text display
        summary = sentiment_data.get('overall_summary', {})
        st.subheader("📊 Sentiment Distribution")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Positive", summary.get('positive_count', 0))
        with col2:
            st.metric("Negative", summary.get('negative_count', 0))
        with col3:
            st.metric("Neutral", summary.get('neutral_count', 0))
        return
    
    summary = sentiment_data.get('overall_summary', {})
    
    labels = ['Positive', 'Negative', 'Neutral']
    values = [
        summary.get('positive_count', 0),
        summary.get('negative_count', 0),
        summary.get('neutral_count', 0)
    ]
    
    colors = ['#2E8B57', '#DC143C', '#808080']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=colors,
        textinfo='label+percent+value',
        textfont_size=12
    )])
    
    fig.update_layout(
        title="Sentiment Distribution",
        title_x=0.5,
        font=dict(size=14),
        showlegend=True,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def extract_video_id(url):
    parsed = urlparse(url)
    if 'youtube.com' in parsed.netloc:
        return parse_qs(parsed.query).get("v", [None])[0]
    elif 'youtu.be' in parsed.netloc:
        return parsed.path.lstrip('/')
    return None
    

def create_theme_chart(themes_data: List[Dict]):
    """Create a bar chart for themes"""
    if not PLOTLY_AVAILABLE or not themes_data:
        # Fallback to simple text display
        st.subheader("🎯 Top Discussion Themes")
        for i, theme in enumerate(themes_data[:5]):
            st.write(f"{i+1}. **{theme['theme']}** - {theme['frequency']} mentions ({theme['sentiment']})")
        return
        
    themes = [theme['theme'] for theme in themes_data[:8]]  # Top 8 themes
    frequencies = [theme['frequency'] for theme in themes_data[:8]]
    sentiments = [theme['sentiment'] for theme in themes_data[:8]]
    
    # Color mapping for sentiments
    color_map = {'Positive': '#2E8B57', 'Negative': '#DC143C', 'Mixed': '#FF8C00'}
    colors = [color_map.get(sentiment, '#808080') for sentiment in sentiments]
    
    fig = go.Figure(data=[go.Bar(
        x=frequencies,
        y=themes,
        orientation='h',
        marker_color=colors,
        text=sentiments,
        textposition='inside'
    )])
    
    fig.update_layout(
        title="Top Discussion Themes",
        xaxis_title="Frequency",
        yaxis_title="Themes",
        height=max(400, len(themes) * 40),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def export_report_to_html(sentiment_data: Dict, insights_data: Dict, video_url: str) -> str:
    """Generate HTML report for download"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>YouTube Comment Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            h1, h2, h3 {{ color: #333; }}
            .summary {{ background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0; }}
            .section {{ margin: 30px 0; }}
            .quote {{ font-style: italic; color: #666; margin: 10px 0; padding: 10px; background: #f9f9f9; border-left: 3px solid #ddd; }}
            .recommendation {{ background: #e8f4f8; padding: 15px; border-left: 4px solid #2E8B57; margin: 10px 0; }}
            .concern {{ background: #fff5f5; padding: 15px; border-left: 4px solid #DC143C; margin: 10px 0; }}
            .loved {{ background: #f0f8f0; padding: 15px; border-left: 4px solid #2E8B57; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>📊 YouTube Comment Analysis Report</h1>
        <p><strong>Video URL:</strong> {video_url}</p>
        <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <h2>📈 Sentiment Summary</h2>
            <p>✅ <strong>Positive:</strong> {sentiment_data.get('overall_summary', {}).get('positive_count', 0)} comments</p>
            <p>❌ <strong>Negative:</strong> {sentiment_data.get('overall_summary', {}).get('negative_count', 0)} comments</p>
            <p>⚪ <strong>Neutral:</strong> {sentiment_data.get('overall_summary', {}).get('neutral_count', 0)} comments</p>
            <p>🎯 <strong>Dominant Sentiment:</strong> {sentiment_data.get('overall_summary', {}).get('dominant_sentiment', 'N/A')}</p>
        </div>
        
        <div class="section">
            <h2>💚 What Audiences Love</h2>
    """
    
    # Add loved elements
    for element in insights_data.get('loved_elements', []):
        html_content += f"""
            <div class="loved">
                <h3>✨ {element['element']}</h3>
                <p><strong>Business Value:</strong> {element['business_value']}</p>
                <div class="quote">"{element['quotes'][0] if element['quotes'] else 'No quotes available'}"</div>
            </div>
        """
    
    html_content += """
        </div>
        
        <div class="section">
            <h2>⚠️ Concerns & Risks</h2>
    """
    
    # Add concerns
    for concern in insights_data.get('concerns_risks', []):
        html_content += f"""
            <div class="concern">
                <h3>🚨 {concern['concern']}</h3>
                <p><strong>Mitigation:</strong> {concern['mitigation']}</p>
                <div class="quote">"{concern['quotes'][0] if concern['quotes'] else 'No quotes available'}"</div>
            </div>
        """
    
    html_content += """
        </div>
        
        <div class="section">
            <h2>🎯 Business Recommendations</h2>
    """
    
    # Add recommendations
    for rec in insights_data.get('business_recommendations', []):
        priority_color = {'High': '#DC143C', 'Medium': '#FF8C00', 'Low': '#2E8B57'}.get(rec['priority'], '#808080')
        priority_emoji = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}.get(rec['priority'], '⚪')
        html_content += f"""
            <div class="recommendation">
                <h3 style="color: {priority_color};">{priority_emoji} {rec['action']} (Priority: {rec['priority']})</h3>
                <p><strong>Rationale:</strong> {rec['rationale']}</p>
            </div>
        """
    
    html_content += """
        </div>
        
        <div class="section">
            <h2>📋 Summary</h2>
            <p>This report was generated using AI analysis of YouTube comments to provide actionable business insights. 
            The analysis focuses on sentiment distribution, key themes, and strategic recommendations for content improvement.</p>
        </div>
    </body>
    </html>
    """
    
    return html_content

def main():
    st.title("📊 YouTube Comment Insight Analyzer")
    st.markdown("**Transform YouTube comments into actionable business intelligence**")
    
     # Dependency check
    missing_deps = []
    if not PLOTLY_AVAILABLE:
        missing_deps.append("plotly")
    if not GROQ_AVAILABLE:
        missing_deps.append("groq")
    
    if missing_deps:
        st.warning(f"⚠️ Optional dependencies missing: {', '.join(missing_deps)}")
        st.info("Install with: `pip install " + " ".join(missing_deps) + "`")
        if not GROQ_AVAILABLE:
            st.error("Groq is required for AI analysis. Please install it to continue.")
        if not PLOTLY_AVAILABLE:
            st.info("Charts will be displayed as text without Plotly.") 
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Debug info
    with st.sidebar.expander("🔧 Debug Info"):
        st.write(f"Streamlit: {st.__version__}")
        st.write(f"Pandas: {pd.__version__}")
        st.write(f"Plotly: {'✅' if PLOTLY_AVAILABLE else '❌'}")
        st.write(f"Groq: {'✅' if GROQ_AVAILABLE else '❌'}")
    
    # API Key input
    groq_api_key = st.sidebar.text_input(
        "Groq API Key",
        type="password",
        help="Enter your Groq API key to enable AI analysis"
    )
    
    if not GROQ_AVAILABLE and groq_api_key:
        st.sidebar.error("Please install Groq: `pip install groq`")
    
    # Video type selection
    video_type = st.sidebar.selectbox(
        "Video Type",
        ["general", "music"],
        help="Select video type for specialized analysis"
    )
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    df = None
    with col1:
        st.subheader("📹 Video Input")
        video_url = st.text_input(
            "YouTube Video URL",
            placeholder="https://www.youtube.com/watch?v=..."
        )
    with col2:
        st.subheader("📂 Upload Comments")
        uploaded_file = st.file_uploader(
            "Upload CSV with comments",
            type=['csv'],
            help="Upload your scraped comments CSV file"
        )

    if video_url:
        video_id = extract_video_id(video_url)

        if video_id:
            try:
                # st.write(video_id)
                comments = get_all_comments(video_id)
                # save_to_csv(comments)

                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=comments[0].keys())
                writer.writeheader()
                writer.writerows(comments)
                output.seek(0)

            # ✅ Load into pandas DataFrame
                uploaded_file_csv=output
                df = pd.read_csv(output)
                st.success(f"✅ Saved {len(comments)} comments to CSV.")
            except Exception as e:
                st.error(f"❌ Error fetching comments: {e}")
        else:
            st.error("❌ Could not extract video ID from URL.")

    
    if uploaded_file is not None:
            # Load comments
            df = pd.read_csv(uploaded_file)


    if df is not None:

        try:          
            # Detect comment column
            comment_columns = ['comment', 'Comment', 'text', 'Text', 'content', 'Content', 'body', 'Body']
            comment_col = None
            for col in comment_columns:
                if col in df.columns:
                    comment_col = col
                    break
            
            if comment_col is None:
                st.error("❌ Could not find comment column. Please ensure your CSV has a column named 'comment', 'text', or 'content'.")
                st.write("**Available columns:**", list(df.columns))
                return
            
            comments = df[comment_col].dropna().astype(str).tolist()
            
            if len(comments) == 0:
                st.error("❌ No comments found in the uploaded file.")
                return
            
            st.success(f"✅ Loaded {len(comments)} comments successfully!")
            
            # Display sample comments
            with st.expander("📝 Sample Comments"):
                for i, comment in enumerate(comments[:5]):
                    st.write(f"{i+1}. {comment}")
            
            # Analysis section
            if GROQ_AVAILABLE and groq_api_key:
                if st.button("🔍 Analyze Comments", type="primary"):
                    analyzer = CommentAnalyzer(groq_api_key)
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Sentiment Analysis
                    status_text.text("🔄 Analyzing sentiment...")
                    progress_bar.progress(25)
                    
                    # Process comments in batches
                    batch_size = len(df)//10
                    all_sentiments = []
                    sentiment_summary = {"positive_count": 0, "negative_count": 0, "neutral_count": 0}
                    
                    # Limit to first 60 comments for faster processing
                    comments_to_analyze = comments[:200]
                    
                    for i in range(0, len(comments_to_analyze), batch_size):
                        batch = comments_to_analyze[i:i+batch_size]
                        sentiment_result = analyzer.analyze_sentiment_batch(batch)
                        
                        if sentiment_result:
                            all_sentiments.extend(sentiment_result.get('sentiments', []))
                            batch_summary = sentiment_result.get('overall_summary', {})
                            sentiment_summary['positive_count'] += batch_summary.get('positive_count', 0)
                            sentiment_summary['negative_count'] += batch_summary.get('negative_count', 0)
                            sentiment_summary['neutral_count'] += batch_summary.get('neutral_count', 0)
                    
                    # Determine dominant sentiment
                    if sentiment_summary['positive_count'] > 0 or sentiment_summary['negative_count'] > 0 or sentiment_summary['neutral_count'] > 0:
                        sentiment_summary['dominant_sentiment'] = max(
                            sentiment_summary, key=sentiment_summary.get
                        ).replace('_count', '').title()
                    else:
                        sentiment_summary['dominant_sentiment'] = 'Unknown'
                    
                    progress_bar.progress(50)
                    
                    # Theme and Insights Analysis
                    status_text.text("🔄 Extracting themes and insights...")
                    progress_bar.progress(75)
                    
                    insights_result = analyzer.extract_themes_and_insights(comments, video_type)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    
                    if insights_result and any(sentiment_summary.values()):
                        # Display results
                        st.header("📊 Analysis Results")
                        
                        # Sentiment visualization
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            sentiment_data = {'overall_summary': sentiment_summary}
                            create_sentiment_chart(sentiment_data)
                        
                        with col2:
                            # Theme visualization
                            if insights_result.get('top_themes'):
                                create_theme_chart(insights_result['top_themes'])
                        
                        # Insights sections
                        st.subheader("💚 What Audiences Love")
                        if insights_result.get('loved_elements'):
                            for element in insights_result['loved_elements']:
                                with st.expander(f"✨ {element['element']}"):
                                    st.write(f"**Business Value:** {element['business_value']}")
                                    if element.get('quotes'):
                                        st.write("**Sample Quotes:**")
                                        for quote in element['quotes'][:3]:
                                            st.write(f"💬 _{quote}_")
                        else:
                            st.info("No specific loved elements identified in the comments.")
                        
                        st.subheader("⚠️ Concerns & Risks")
                        if insights_result.get('concerns_risks'):
                            for concern in insights_result['concerns_risks']:
                                with st.expander(f"🚨 {concern['concern']}"):
                                    st.write(f"**Mitigation Strategy:** {concern['mitigation']}")
                                    if concern.get('quotes'):
                                        st.write("**Sample Quotes:**")
                                        for quote in concern['quotes'][:3]:
                                            st.write(f"💬 _{quote}_")
                        else:
                            st.info("No major concerns identified in the comments.")
                        
                        st.subheader("🎯 Business Recommendations")
                        if insights_result.get('business_recommendations'):
                            for rec in insights_result['business_recommendations']:
                                priority_emoji = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}.get(rec['priority'], '⚪')
                                with st.expander(f"{priority_emoji} {rec['action']} (Priority: {rec['priority']})"):
                                    st.write(f"**Rationale:** {rec['rationale']}")
                        else:
                            st.info("No specific business recommendations generated.")
                        
                        # Export functionality
                        st.subheader("📄 Export Report")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # HTML Report
                            html_report = export_report_to_html(sentiment_data, insights_result, video_url or "N/A")
                            st.download_button(
                                label="📄 Download HTML Report",
                                data=html_report,
                                file_name=f"youtube_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html",
                                mime="text/html"
                            )
                        
                        with col2:
                            # CSV Export
                            export_data = []
                            for element in insights_result.get('loved_elements', []):
                                export_data.append({
                                    'Type': 'Loved Element',
                                    'Item': element['element'],
                                    'Details': element['business_value']
                                })
                            for concern in insights_result.get('concerns_risks', []):
                                export_data.append({
                                    'Type': 'Concern',
                                    'Item': concern['concern'],
                                    'Details': concern['mitigation']
                                })
                            for rec in insights_result.get('business_recommendations', []):
                                export_data.append({
                                    'Type': 'Recommendation',
                                    'Item': rec['action'],
                                    'Details': f"Priority: {rec['priority']} - {rec['rationale']}"
                                })
                            
                            if export_data:
                                insights_df = pd.DataFrame(export_data)
                                csv_buffer = io.StringIO()
                                insights_df.to_csv(csv_buffer, index=False)
                                
                                st.download_button(
                                    label="📊 Download CSV Data",
                                    data=csv_buffer.getvalue(),
                                    file_name=f"youtube_insights_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                    
                    else:
                        st.error("❌ Failed to analyze comments. Please check your API key and try again.")
            
            elif not GROQ_AVAILABLE:
                st.error("❌ Groq library not installed. Please install it with: `pip install groq`")
            
            elif not groq_api_key:
                st.info("👆 Please enter your Groq API key in the sidebar to begin analysis.")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())
    
    else:
        # Instructions when no file is uploaded
        st.info("📁 Please upload your comments CSV file to begin analysis.")
        
        with st.expander("📋 CSV Format Requirements"):
            st.markdown("""
            Your CSV file should contain:
            - A column named 'comment', 'text', 'content', or similar
            - One comment per row
            - UTF-8 encoding for special characters
            
            **Example CSV format:**
            ```
            comment
            "Great video! Love the music!"
            "Not my favorite, but still good"
            "Amazing production quality"
            ```
            """)
        
        with st.expander("🔑 How to get Groq API Key"):
            st.markdown("""
            1. Visit [Groq Console](https://console.groq.com)
            2. Sign up or log in to your account
            3. Navigate to API Keys section
            4. Create a new API key
            5. Copy and paste it in the sidebar
            
            **Note:** Groq offers free tier with generous limits for testing.
            """)

    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit, Pandas, and Groq | "
        "Powered by LLaMA-3.1 for intelligent comment analysis"
    )

if __name__ == "__main__":
    main()