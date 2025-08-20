"""
Media Mix Modeling Streamlit Demo Application

Interactive web application for demonstrating MMM capabilities including:
- Attribution analysis
- Budget optimization  
- Cross-channel synergy analysis
- Real-time predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import MMM components
try:
    from src.attribution.attribution_engine import AttributionEngine
    from src.optimization.budget_optimizer import BudgetOptimizer
    from src.reports.executive_reporter import ExecutiveReporter
except ImportError:
    st.error("Could not import MMM components. Please ensure you're running from the project root.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Media Mix Modeling Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def load_sample_data():
    """Load sample marketing data for the demo."""
    try:
        # Try to load real sample data
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                'data', 'samples', 'campaign_budget_data.csv')
        if os.path.exists(data_path):
            return pd.read_csv(data_path)
        else:
            # Generate synthetic data if sample data not found
            return generate_sample_data()
    except Exception:
        return generate_sample_data()

def generate_sample_data():
    """Generate synthetic marketing data for demonstration."""
    np.random.seed(42)
    
    # Date range
    start_date = datetime.now() - timedelta(days=365)
    dates = pd.date_range(start=start_date, periods=365, freq='D')
    
    channels = ['Search', 'Social Media', 'Display', 'Video', 'Email', 'TV']
    
    data = []
    for date in dates:
        for channel in channels:
            # Base spend with seasonality and trends
            base_spend = np.random.uniform(1000, 5000)
            seasonality = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
            trend = 1 + 0.001 * (date - start_date).days
            
            spend = base_spend * seasonality * trend * np.random.uniform(0.8, 1.2)
            impressions = spend * np.random.uniform(50, 200)
            clicks = impressions * np.random.uniform(0.01, 0.05)
            conversions = clicks * np.random.uniform(0.02, 0.08)
            
            data.append({
                'date': date,
                'channel': channel,
                'spend': round(spend, 2),
                'impressions': int(impressions),
                'clicks': int(clicks),
                'conversions': int(conversions)
            })
    
    return pd.DataFrame(data)

def create_attribution_analysis(data):
    """Perform attribution analysis on the data."""
    # Simulate attribution results
    channels = data['channel'].unique()
    
    attribution_results = {}
    for channel in channels:
        channel_data = data[data['channel'] == channel]
        total_spend = channel_data['spend'].sum()
        total_conversions = channel_data['conversions'].sum()
        
        attribution_results[channel] = {
            'spend': total_spend,
            'conversions': total_conversions,
            'roas': total_conversions * 50 / total_spend if total_spend > 0 else 0,  # Assuming $50 per conversion
            'attribution_score': np.random.uniform(0.1, 0.3)
        }
    
    return attribution_results

def create_optimization_recommendations(attribution_results, total_budget):
    """Create budget optimization recommendations."""
    total_current_spend = sum(result['spend'] for result in attribution_results.values())
    
    recommendations = {}
    for channel, metrics in attribution_results.items():
        current_allocation = metrics['spend'] / total_current_spend
        roas = metrics['roas']
        
        # Simple optimization: allocate more to high-ROAS channels
        if roas > 2.0:
            recommended_allocation = current_allocation * 1.2
        elif roas > 1.5:
            recommended_allocation = current_allocation * 1.1
        elif roas > 1.0:
            recommended_allocation = current_allocation
        else:
            recommended_allocation = current_allocation * 0.8
            
        recommendations[channel] = {
            'current_allocation': current_allocation,
            'recommended_allocation': recommended_allocation,
            'recommended_budget': recommended_allocation * total_budget,
            'expected_lift': (recommended_allocation / current_allocation - 1) * 100 if current_allocation > 0 else 0
        }
    
    # Normalize recommendations to sum to 1
    total_recommended = sum(r['recommended_allocation'] for r in recommendations.values())
    for channel in recommendations:
        recommendations[channel]['recommended_allocation'] /= total_recommended
        recommendations[channel]['recommended_budget'] = recommendations[channel]['recommended_allocation'] * total_budget
    
    return recommendations

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<div class="main-header">📊 Media Mix Modeling Platform</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the **Media Mix Modeling Platform** - your comprehensive solution for marketing attribution, 
    budget optimization, and cross-channel performance analysis.
    """)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🏠 Overview", "📊 Attribution Analysis", "💰 Budget Optimization", 
         "🔄 Cross-Channel Synergy", "📈 Real-time Dashboard", "⚙️ Settings"]
    )
    
    # Load data
    with st.spinner("Loading marketing data..."):
        data = load_sample_data()
    
    if page == "🏠 Overview":
        show_overview(data)
    elif page == "📊 Attribution Analysis":
        show_attribution_analysis(data)
    elif page == "💰 Budget Optimization":
        show_budget_optimization(data)
    elif page == "🔄 Cross-Channel Synergy":
        show_cross_channel_analysis(data)
    elif page == "📈 Real-time Dashboard":
        show_realtime_dashboard(data)
    elif page == "⚙️ Settings":
        show_settings()

def show_overview(data):
    """Show platform overview and key metrics."""
    st.header("Platform Overview")
    
    # Key metrics
    total_spend = data['spend'].sum()
    total_conversions = data['conversions'].sum()
    total_impressions = data['impressions'].sum()
    avg_roas = (total_conversions * 50) / total_spend if total_spend > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Ad Spend", f"${total_spend:,.0f}", "+12.5%")
    with col2:
        st.metric("Total Conversions", f"{total_conversions:,.0f}", "+8.3%")
    with col3:
        st.metric("Total Impressions", f"{total_impressions:,.0f}", "+15.7%")
    with col4:
        st.metric("Average ROAS", f"{avg_roas:.2f}x", "+5.2%")
    
    # Channel performance overview
    st.subheader("Channel Performance Overview")
    
    channel_summary = data.groupby('channel').agg({
        'spend': 'sum',
        'conversions': 'sum',
        'impressions': 'sum'
    }).reset_index()
    
    channel_summary['roas'] = (channel_summary['conversions'] * 50) / channel_summary['spend']
    channel_summary = channel_summary.sort_values('spend', ascending=False)
    
    # Spend by channel
    col1, col2 = st.columns(2)
    
    with col1:
        fig_spend = px.pie(channel_summary, values='spend', names='channel', 
                          title="Ad Spend by Channel")
        st.plotly_chart(fig_spend, use_container_width=True)
    
    with col2:
        fig_roas = px.bar(channel_summary, x='channel', y='roas',
                         title="ROAS by Channel", color='roas',
                         color_continuous_scale='viridis')
        st.plotly_chart(fig_roas, use_container_width=True)
    
    # Platform capabilities
    st.subheader("Platform Capabilities")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Attribution Analysis
        - Multi-touch attribution modeling
        - Cross-channel impact measurement
        - Statistical significance testing
        - Custom attribution windows
        """)
    
    with col2:
        st.markdown("""
        ### 💰 Budget Optimization
        - ROI-driven budget allocation
        - Constraint-based optimization
        - Scenario planning
        - Performance forecasting
        """)
    
    with col3:
        st.markdown("""
        ### 🔄 Advanced Analytics
        - Cross-channel synergy analysis
        - Incrementality testing
        - Media saturation curves
        - Competitive intelligence
        """)

def show_attribution_analysis(data):
    """Show attribution analysis interface."""
    st.header("Attribution Analysis")
    
    # Attribution settings
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Analysis Settings")
        
        attribution_model = st.selectbox(
            "Attribution Model:",
            ["Last Touch", "First Touch", "Linear", "Time Decay", "Data Driven"]
        )
        
        time_window = st.slider("Attribution Window (days):", 1, 90, 30)
        
        selected_channels = st.multiselect(
            "Channels to analyze:",
            data['channel'].unique(),
            default=data['channel'].unique()
        )
        
        if st.button("Run Attribution Analysis"):
            with st.spinner("Running attribution analysis..."):
                # Perform analysis
                filtered_data = data[data['channel'].isin(selected_channels)]
                attribution_results = create_attribution_analysis(filtered_data)
                st.session_state.attribution_results = attribution_results
    
    with col2:
        st.subheader("Attribution Results")
        
        if 'attribution_results' in st.session_state:
            results = st.session_state.attribution_results
            
            # Create attribution dataframe
            attribution_df = pd.DataFrame.from_dict(results, orient='index')
            attribution_df = attribution_df.reset_index().rename(columns={'index': 'channel'})
            
            # Attribution scores visualization
            fig = px.bar(attribution_df, x='channel', y='attribution_score',
                        title=f"{attribution_model} Attribution Scores",
                        color='attribution_score',
                        color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
            
            # Results table
            st.subheader("Detailed Results")
            
            display_df = attribution_df.copy()
            display_df['spend'] = display_df['spend'].apply(lambda x: f"${x:,.0f}")
            display_df['conversions'] = display_df['conversions'].apply(lambda x: f"{x:,.0f}")
            display_df['roas'] = display_df['roas'].apply(lambda x: f"{x:.2f}x")
            display_df['attribution_score'] = display_df['attribution_score'].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(display_df)

def show_budget_optimization(data):
    """Show budget optimization interface."""
    st.header("Budget Optimization")
    
    # Get attribution results or create them
    if 'attribution_results' not in st.session_state:
        st.session_state.attribution_results = create_attribution_analysis(data)
    
    attribution_results = st.session_state.attribution_results
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Optimization Settings")
        
        current_budget = sum(result['spend'] for result in attribution_results.values())
        
        new_budget = st.number_input(
            "Total Budget ($):",
            value=int(current_budget),
            min_value=1000,
            step=1000
        )
        
        optimization_goal = st.selectbox(
            "Optimization Goal:",
            ["Maximize ROAS", "Maximize Conversions", "Minimize Cost per Acquisition"]
        )
        
        constraints = st.multiselect(
            "Budget Constraints:",
            ["Minimum 10% per channel", "Maximum 40% per channel", "Maintain current ratios"]
        )
        
        if st.button("Optimize Budget"):
            with st.spinner("Optimizing budget allocation..."):
                recommendations = create_optimization_recommendations(attribution_results, new_budget)
                st.session_state.optimization_results = recommendations
    
    with col2:
        st.subheader("Optimization Results")
        
        if 'optimization_results' in st.session_state:
            recommendations = st.session_state.optimization_results
            
            # Create comparison chart
            channels = list(recommendations.keys())
            current_budgets = [attribution_results[ch]['spend'] for ch in channels]
            recommended_budgets = [recommendations[ch]['recommended_budget'] for ch in channels]
            
            fig = go.Figure(data=[
                go.Bar(name='Current', x=channels, y=current_budgets),
                go.Bar(name='Recommended', x=channels, y=recommended_budgets)
            ])
            fig.update_layout(barmode='group', title='Current vs Recommended Budget Allocation')
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations table
            st.subheader("Budget Recommendations")
            
            rec_df = pd.DataFrame.from_dict(recommendations, orient='index')
            rec_df = rec_df.reset_index().rename(columns={'index': 'channel'})
            
            # Format for display
            display_rec = rec_df.copy()
            display_rec['current_allocation'] = display_rec['current_allocation'].apply(lambda x: f"{x:.1%}")
            display_rec['recommended_allocation'] = display_rec['recommended_allocation'].apply(lambda x: f"{x:.1%}")
            display_rec['recommended_budget'] = display_rec['recommended_budget'].apply(lambda x: f"${x:,.0f}")
            display_rec['expected_lift'] = display_rec['expected_lift'].apply(lambda x: f"{x:+.1f}%")
            
            st.dataframe(display_rec)
            
            # Key insights
            total_lift = sum(rec['expected_lift'] for rec in recommendations.values() if rec['expected_lift'] > 0)
            st.markdown(f"""
            <div class="insight-box">
                <h4>💡 Key Insights</h4>
                <ul>
                    <li>Expected overall performance lift: <strong>{total_lift:.1f}%</strong></li>
                    <li>Budget reallocation recommended for <strong>{len([r for r in recommendations.values() if abs(r['expected_lift']) > 5])} channels</strong></li>
                    <li>Optimization goal: <strong>{optimization_goal}</strong></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

def show_cross_channel_analysis(data):
    """Show cross-channel synergy analysis."""
    st.header("Cross-Channel Synergy Analysis")
    
    st.markdown("""
    Analyze how different marketing channels work together to drive conversions.
    This analysis identifies channel combinations that create synergistic effects.
    """)
    
    # Synergy matrix (simulated)
    channels = data['channel'].unique()
    synergy_matrix = np.random.uniform(0.8, 1.3, size=(len(channels), len(channels)))
    np.fill_diagonal(synergy_matrix, 1.0)
    
    # Make matrix symmetric
    synergy_matrix = (synergy_matrix + synergy_matrix.T) / 2
    np.fill_diagonal(synergy_matrix, 1.0)
    
    # Create heatmap
    fig = px.imshow(synergy_matrix, 
                    x=channels, 
                    y=channels,
                    title="Channel Synergy Matrix",
                    color_continuous_scale='RdYlGn',
                    aspect="auto")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top synergies
    st.subheader("Top Channel Combinations")
    
    synergies = []
    for i in range(len(channels)):
        for j in range(i+1, len(channels)):
            synergies.append({
                'channel_1': channels[i],
                'channel_2': channels[j],
                'synergy_score': synergy_matrix[i, j],
                'lift': (synergy_matrix[i, j] - 1) * 100
            })
    
    synergies_df = pd.DataFrame(synergies)
    synergies_df = synergies_df.sort_values('synergy_score', ascending=False).head(10)
    
    for _, row in synergies_df.iterrows():
        lift_color = "green" if row['lift'] > 0 else "red"
        st.markdown(f"""
        **{row['channel_1']} + {row['channel_2']}**: 
        <span style="color: {lift_color}; font-weight: bold;">{row['lift']:+.1f}% lift</span>
        """, unsafe_allow_html=True)

def show_realtime_dashboard(data):
    """Show real-time performance dashboard."""
    st.header("Real-time Performance Dashboard")
    
    # Auto-refresh option
    auto_refresh = st.checkbox("Auto-refresh (30s)")
    if auto_refresh:
        st.rerun()
    
    # Today's performance (simulated)
    today = datetime.now().date()
    
    # Key metrics for today
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Today's Spend", "$12,450", "+8.2%")
    with col2:
        st.metric("Today's Conversions", "187", "+12.5%")
    with col3:
        st.metric("Current ROAS", "2.34x", "+0.15")
    with col4:
        st.metric("Active Campaigns", "23", "+2")
    
    # Real-time channel performance
    st.subheader("Channel Performance - Last 24 Hours")
    
    # Generate hourly data for the last 24 hours
    hours = pd.date_range(end=datetime.now(), periods=24, freq='H')
    
    realtime_data = []
    for hour in hours:
        for channel in data['channel'].unique()[:4]:  # Show top 4 channels
            value = np.random.uniform(50, 200) * (1 + 0.1 * np.sin(2 * np.pi * hour.hour / 24))
            realtime_data.append({
                'hour': hour,
                'channel': channel,
                'conversions': int(value)
            })
    
    realtime_df = pd.DataFrame(realtime_data)
    
    fig = px.line(realtime_df, x='hour', y='conversions', color='channel',
                  title="Conversions by Channel - Last 24 Hours")
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance alerts
    st.subheader("Performance Alerts")
    
    alerts = [
        {"type": "warning", "message": "Search campaign CTR dropped 15% in the last hour", "time": "2 minutes ago"},
        {"type": "success", "message": "Social Media ROAS exceeded target by 20%", "time": "15 minutes ago"},
        {"type": "info", "message": "Display campaign budget 80% spent for today", "time": "1 hour ago"},
    ]
    
    for alert in alerts:
        if alert["type"] == "warning":
            st.warning(f"⚠️ {alert['message']} ({alert['time']})")
        elif alert["type"] == "success":
            st.success(f"✅ {alert['message']} ({alert['time']})")
        else:
            st.info(f"ℹ️ {alert['message']} ({alert['time']})")

def show_settings():
    """Show application settings."""
    st.header("Settings")
    
    st.subheader("Data Sources")
    data_sources = st.multiselect(
        "Connected data sources:",
        ["Google Ads", "Facebook Ads", "Google Analytics", "Salesforce", "HubSpot"],
        default=["Google Ads", "Facebook Ads"]
    )
    
    st.subheader("Attribution Settings")
    default_attribution = st.selectbox(
        "Default attribution model:",
        ["Data Driven", "Last Touch", "First Touch", "Linear", "Time Decay"]
    )
    
    default_window = st.slider("Default attribution window (days):", 1, 90, 30)
    
    st.subheader("Notifications")
    email_alerts = st.checkbox("Email performance alerts", value=True)
    slack_notifications = st.checkbox("Slack notifications", value=False)
    
    st.subheader("Export Options")
    export_format = st.selectbox("Preferred export format:", ["CSV", "Excel", "JSON"])
    
    if st.button("Save Settings"):
        st.success("Settings saved successfully!")

if __name__ == "__main__":
    main()