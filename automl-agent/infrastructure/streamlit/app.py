"""
AutoML Agent Platform - Streamlit Web Interface

Revolutionary multi-agent AutoML system with interactive web interface.
Showcases collaborative intelligence between specialized ML agents.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
from datetime import datetime
from typing import Dict, Any, List
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from src.pipelines.data_pipeline import DataProcessingPipeline
    from src.agents.eda_agent import EDAAgent
    from src.agents.classification_agent import ClassificationAgent
    from src.agents.qa_agent import QualityAssuranceAgent
    from src.agents.base_agent import TaskContext
    from data.samples.dataset_loader import RealDatasetLoader
    PLATFORM_AVAILABLE = True
except ImportError as e:
    st.error(f"Import error: {e}")
    PLATFORM_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="AutoML Agent Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS using Joshua Morrison website styling
st.markdown("""
<style>
    /* Import website color scheme and typography */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .subtitle {
        font-size: 1.5rem;
        font-weight: 300;
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    
    .agent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 0.8rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    
    .agent-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
    }
    
    .feature-card {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
        border: 1px solid #e9ecef;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #667eea;
        margin: 0.8rem 0;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
    }
    
    .btn-primary {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 15px 30px;
        font-weight: 600;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .btn-primary:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
    }
    
    .status-running {
        color: #28a745;
        font-weight: 600;
    }
    
    .status-completed {
        color: #667eea;
        font-weight: 600;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: 600;
    }
    
    .highlight {
        color: #667eea;
        font-weight: 600;
    }
    
    .section-title {
        font-size: 2.5rem;
        margin-bottom: 2rem;
        color: #333;
        position: relative;
        text-align: center;
    }
    
    .section-title::after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 50%;
        transform: translateX(-50%);
        width: 80px;
        height: 4px;
        background: linear-gradient(45deg, #667eea, #764ba2);
    }
    
    /* Streamlit specific overrides */
    .stSelectbox > div > div {
        border-radius: 10px;
    }
    
    .stButton > button {
        border-radius: 25px;
        border: none;
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    .floating-element {
        position: fixed;
        background: rgba(102, 126, 234, 0.1);
        border-radius: 50%;
        animation: float 6s ease-in-out infinite;
        z-index: -1;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Header with website styling
    st.markdown('<h1 class="main-header">AutoML Agent Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Revolutionary Multi-Agent AutoML with Collaborative Intelligence</p>', unsafe_allow_html=True)
    
    # Add floating elements for visual appeal
    st.markdown("""
    <div class="floating-element" style="width: 80px; height: 80px; top: 20%; left: 10%;"></div>
    <div class="floating-element" style="width: 60px; height: 60px; top: 60%; right: 10%; animation-delay: 2s;"></div>
    <div class="floating-element" style="width: 40px; height: 40px; top: 80%; left: 20%; animation-delay: 4s;"></div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🎛️ Control Panel")
    page = st.sidebar.selectbox(
        "Choose Your Experience",
        ["🏠 Platform Overview", "🚀 Quick Start Demo", "🎯 Custom Task", "📊 Agent Analytics", "🔬 Agent Communication"]
    )
    
    if page == "🏠 Platform Overview":
        show_platform_overview()
    elif page == "🚀 Quick Start Demo":
        show_quick_start_demo()
    elif page == "🎯 Custom Task":
        show_custom_task()
    elif page == "📊 Agent Analytics":
        show_agent_analytics()
    elif page == "🔬 Agent Communication":
        show_agent_communication()

def show_platform_overview():
    """Show platform overview and architecture."""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="section-title">Revolutionary AutoML Architecture</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <p><strong>Traditional AutoML</strong>: Monolithic systems that try to do everything</p>
            <p><strong class="highlight">Our Innovation</strong>: Specialized AI agents that collaborate intelligently until quality thresholds are met</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Architecture diagram
        create_architecture_diagram()
        
        st.markdown('<h2 class="section-title">Key Innovations</h2>', unsafe_allow_html=True)
        
        features = [
            ("🧠 Collaborative Intelligence", "Agents share knowledge and refine each other's work"),
            ("🎨 Quality-Driven Refinement", "Iterative improvement until performance targets met"),
            ("🗣️ Natural Language Interface", "Describe your ML task in plain English"),
            ("⚡ Specialized Expertise", "Each agent excels in their specific domain"),
            ("🔄 Human-in-the-Loop", "Expert oversight and approval at key decision points"),
            ("🚀 Production Ready", "FastAPI endpoints, monitoring, and deployment")
        ]
        
        for title, desc in features:
            st.markdown(f"""
            <div class="feature-card">
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h2 class="section-title">Available Agents</h2>', unsafe_allow_html=True)
        
        agents = [
            ("🔍 EDA Agent", "Exploratory Data Analysis"),
            ("🧹 Data Hygiene Agent", "Data Cleaning & Preprocessing"),
            ("⚙️ Feature Engineering Agent", "Automated Feature Creation"),
            ("📈 Classification Agent", "Supervised Classification"),
            ("📊 Regression Agent", "Continuous Prediction"),
            ("📝 NLP Agent", "Text Processing & Analysis"),
            ("👁️ Computer Vision Agent", "Image Analysis & CNN"),
            ("⏰ Time Series Agent", "Temporal Data & Forecasting"),
            ("🎯 Router Agent", "Task Analysis & Routing")
        ]
        
        for name, desc in agents:
            st.markdown(f"""
            <div class="agent-card">
                <strong>{name}</strong><br>
                <small>{desc}</small>
            </div>
            """, unsafe_allow_html=True)

def create_architecture_diagram():
    """Create interactive architecture diagram."""
    
    # Create network-style diagram using plotly
    fig = go.Figure()
    
    # Node positions
    nodes = {
        "User Input": (0, 0, "#FF6B6B"),
        "Router Agent": (2, 0, "#00D2D3"),
        "EDA Agent": (4, 2, "#FF6B6B"),
        "Data Hygiene": (4, 1, "#4ECDC4"),
        "Feature Eng": (4, 0, "#45B7D1"),
        "ML Agents": (4, -1, "#96CEB4"),
        "Results": (6, 0, "#FECA57")
    }
    
    # Add nodes
    for name, (x, y, color) in nodes.items():
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=30, color=color),
            text=name,
            textposition="middle center",
            textfont=dict(color="white", size=10),
            showlegend=False
        ))
    
    # Add edges
    edges = [
        ("User Input", "Router Agent"),
        ("Router Agent", "EDA Agent"),
        ("Router Agent", "Data Hygiene"),
        ("Router Agent", "Feature Eng"),
        ("Router Agent", "ML Agents"),
        ("EDA Agent", "Results"),
        ("Data Hygiene", "Results"),
        ("Feature Eng", "Results"),
        ("ML Agents", "Results")
    ]
    
    for start, end in edges:
        x0, y0, _ = nodes[start]
        x1, y1, _ = nodes[end]
        
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines',
            line=dict(color="rgba(100,100,100,0.3)", width=2),
            showlegend=False
        ))
    
    fig.update_layout(
        title="🏗️ Multi-Agent Architecture Flow",
        showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
        height=300,
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_quick_start_demo():
    """Show quick start demo with real AutoML workflow."""
    
    st.markdown("## 🚀 Real AutoML Demo")
    st.markdown("**Experience real machine learning with actual data and trained models!**")
    
    if not PLATFORM_AVAILABLE:
        st.error("❌ AutoML platform not available. Please check your installation.")
        return
    
    # Demo dataset selection
    demo_options = {
        "📈 Customer Churn Prediction": {
            "task": "Predict customer churn using real Telco dataset",
            "data_type": "classification",
            "description": "7,032 customers • 21 features • Real business data"
        },
        "🏠 House Price Prediction": {
            "task": "Predict California housing prices",
            "data_type": "regression", 
            "description": "20,640 houses • 8 features • Real estate data"
        }
    }
    
    selected_demo = st.selectbox(
        "Choose a real ML demo:",
        list(demo_options.keys()),
        help="Select a demo to run with real data and ML models"
    )
    
    demo_info = demo_options[selected_demo]
    
    # Show demo information
    st.markdown(f"""
    <div class="feature-card">
        <h3>{selected_demo}</h3>
        <p><strong>Task:</strong> {demo_info['task']}</p>
        <p><strong>Type:</strong> {demo_info['data_type']}</p>
        <p><strong>Dataset:</strong> {demo_info['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'demo_results' not in st.session_state:
        st.session_state.demo_results = None
    
    # Run demo button
    if st.button("🚀 Run Real AutoML Demo", type="primary"):
        run_real_automl_demo(demo_info['data_type'])
    
    # Display results if available
    if st.session_state.demo_results:
        display_demo_results(st.session_state.demo_results)

def run_real_automl_demo(task_type: str):
    """Run the real AutoML workflow."""
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Load real data
        status_text.text("Loading real dataset...")
        progress_bar.progress(20)
        
        loader = RealDatasetLoader()
        if task_type == "classification":
            df, target_column = loader.load_customer_churn()
        else:
            df, target_column = loader.load_california_housing()
        
        st.success(f"✅ Loaded real dataset: {df.shape[0]} samples, {df.shape[1]} features")
        
        # Step 2: Data processing pipeline
        status_text.text("Processing data through pipeline...")
        progress_bar.progress(40)
        
        pipeline = DataProcessingPipeline()
        pipeline_result = pipeline.process_dataset(df, target_column, task_type)
        
        if not pipeline_result.success:
            st.error("❌ Data processing failed!")
            return
        
        # Step 3: EDA Agent
        status_text.text("Running EDA Agent...")
        progress_bar.progress(60)
        
        eda_agent = EDAAgent()
        eda_context = TaskContext(
            task_id="demo_eda",
            user_input=f"Analyze data for {task_type}"
        )
        eda_context.data = pipeline_result.data
        eda_context.target_column = target_column
        
        eda_result = eda_agent.execute_task(eda_context)
        
        # Step 4: Model training (if classification)
        model_result = None
        if task_type == "classification":
            status_text.text("Training ML models...")
            progress_bar.progress(80)
            
            class_agent = ClassificationAgent()
            class_context = TaskContext(
                task_id="demo_classification",
                user_input="Train classification models"
            )
            class_context.splits = pipeline_result.metadata['splits']
            
            model_result = class_agent.execute_task(class_context)
        
        # Step 5: Quality Assurance validation
        status_text.text("Running Quality Assurance validation...")
        progress_bar.progress(90)
        
        qa_agent = QualityAssuranceAgent()
        qa_context = TaskContext(
            task_id="demo_qa",
            user_input=f"Validate quality of AutoML workflow for {task_type}"
        )
        qa_context.data = pipeline_result.data
        qa_context.target_column = target_column
        qa_context.eda_result = eda_result if eda_result.success else None
        qa_context.model_result = model_result if model_result and model_result.success else None
        
        qa_result = qa_agent.execute_task(qa_context)
        
        progress_bar.progress(100)
        status_text.text("✅ AutoML demo completed!")
        
        # Store results
        st.session_state.demo_results = {
            "pipeline_result": pipeline_result,
            "eda_result": eda_result,
            "model_result": model_result,
            "qa_result": qa_result,
            "task_type": task_type,
            "target_column": target_column
        }
        
        st.balloons()  # Celebration!
        
    except Exception as e:
        st.error(f"❌ Demo failed: {e}")

def display_demo_results(results):
    """Display comprehensive demo results."""
    
    st.markdown("## 📊 Demo Results")
    
    pipeline_result = results["pipeline_result"]
    eda_result = results["eda_result"]
    model_result = results["model_result"]
    qa_result = results.get("qa_result")
    
    # Summary metrics with QA
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Processing Time", f"{pipeline_result.processing_time:.2f}s")
    with col2:
        quality_score = eda_result.data["data_profile"]["quality_score"] if eda_result.success else 0
        st.metric("Data Quality", f"{quality_score:.3f}")
    with col3:
        if model_result and model_result.success:
            best_score = model_result.data["best_score"]
            st.metric("Best Model Score", f"{best_score:.4f}")
        else:
            st.metric("Model Score", "N/A")
    with col4:
        splits = pipeline_result.metadata["data_splits"]
        st.metric("Training Samples", splits["train_size"])
    with col5:
        if qa_result and qa_result.success:
            overall_quality = qa_result.data["overall_quality_score"]
            st.metric("QA Score", f"{overall_quality:.3f}")
        else:
            st.metric("QA Score", "N/A")
    
    # Detailed results tabs with QA
    tab1, tab2, tab3, tab4 = st.tabs(["📈 EDA Results", "🤖 Model Results", "🔍 Quality Assurance", "📋 Summary"])
    
    with tab1:
        if eda_result and eda_result.success:
            st.subheader("Data Analysis")
            
            # Data quality
            quality = eda_result.data["data_profile"]["quality_score"]
            st.write(f"**Data Quality Score:** {quality:.3f}/1.0")
            
            # Visualizations
            viz_count = len(eda_result.data["visualizations"])
            st.write(f"**Visualizations Generated:** {viz_count}")
            
            # Recommendations
            if eda_result.recommendations:
                st.subheader("Key Insights")
                for i, rec in enumerate(eda_result.recommendations, 1):
                    st.write(f"{i}. {rec}")
        else:
            st.warning("EDA results not available")
    
    with tab2:
        if model_result and model_result.success:
            st.subheader("Machine Learning Results")
            
            best_model = model_result.data["best_model"]
            best_score = model_result.data["best_score"]
            models_trained = len(model_result.data["model_results"])
            
            st.write(f"**Best Model:** {best_model}")
            st.write(f"**Accuracy:** {best_score:.4f}")
            st.write(f"**Models Trained:** {models_trained}")
            
            # Model comparison chart
            if "evaluation_report" in model_result.data:
                rankings = model_result.data["evaluation_report"].model_rankings
                if rankings:
                    ranking_df = pd.DataFrame(rankings)
                    
                    fig = px.bar(
                        ranking_df,
                        x="model_name",
                        y="validation_score", 
                        title="Model Performance Comparison",
                        color="validation_score"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Model results not available")
    
    with tab3:
        if qa_result and qa_result.success:
            st.subheader("Quality Assurance Validation")
            
            qa_data = qa_result.data
            overall_score = qa_data["overall_quality_score"]
            quality_level = qa_data["quality_level"]
            
            # Overall quality display
            st.write(f"**Overall Quality Score:** {overall_score:.3f}/1.0")
            st.write(f"**Quality Level:** {quality_level.title()}")
            
            # Quality level color coding
            if overall_score >= 0.8:
                st.success(f"🎉 Excellent quality workflow!")
            elif overall_score >= 0.7:
                st.info(f"👍 Good quality workflow")
            elif overall_score >= 0.6:
                st.warning(f"⚠️ Acceptable quality - some improvements possible")
            else:
                st.error(f"❌ Quality concerns - review recommendations")
            
            # Dimension scores
            st.subheader("Quality Dimensions")
            dimension_scores = qa_data["dimension_scores"]
            
            col1, col2 = st.columns(2)
            with col1:
                for i, (dimension, score) in enumerate(list(dimension_scores.items())[:3]):
                    st.metric(dimension.replace("_", " ").title(), f"{score:.3f}")
            with col2:
                for dimension, score in list(dimension_scores.items())[3:]:
                    st.metric(dimension.replace("_", " ").title(), f"{score:.3f}")
            
            # Validation summary
            st.subheader("Validation Summary")
            validation_summary = qa_data["validation_summary"]
            
            for category, checks in validation_summary.items():
                passed = sum(1 for check in checks if check.get('passed', False))
                total = len(checks)
                
                if passed == total:
                    st.success(f"✅ {category.replace('_', ' ').title()}: {passed}/{total} checks passed")
                elif passed > total * 0.7:
                    st.info(f"🔵 {category.replace('_', ' ').title()}: {passed}/{total} checks passed")
                else:
                    st.warning(f"⚠️ {category.replace('_', ' ').title()}: {passed}/{total} checks passed")
            
            # Recommendations
            if qa_result.recommendations:
                st.subheader("Quality Recommendations")
                for i, rec in enumerate(qa_result.recommendations[:5], 1):
                    st.write(f"{i}. {rec}")
        else:
            st.warning("Quality assurance results not available")
    
    with tab4:
        st.subheader("Complete Workflow Summary")
        
        st.write("### Pipeline Success ✅")
        st.write(f"- Original shape: {pipeline_result.metadata['original_shape']}")
        st.write(f"- Processed shape: {pipeline_result.metadata['processed_shape']}")
        st.write(f"- Processing time: {pipeline_result.processing_time:.2f} seconds")
        
        if eda_result and eda_result.success:
            st.write("### EDA Analysis ✅")
            st.write(f"- Data quality: {eda_result.data['data_profile']['quality_score']:.3f}")
            st.write(f"- Visualizations: {len(eda_result.data['visualizations'])}")
        
        if model_result and model_result.success:
            st.write("### Model Training ✅")
            st.write(f"- Best model: {model_result.data['best_model']}")
            st.write(f"- Accuracy: {model_result.data['best_score']:.4f}")
        
        if qa_result and qa_result.success:
            st.write("### Quality Assurance ✅")
            st.write(f"- Overall quality: {qa_result.data['overall_quality_score']:.3f}")
            st.write(f"- Quality level: {qa_result.data['quality_level']}")
            st.write(f"- Recommendations: {len(qa_result.recommendations)}")
        
        st.markdown("**🎉 Real AutoML workflow with quality validation completed successfully!**")

def show_custom_task():
    """Allow users to create custom AutoML tasks."""
    st.header("🎯 Custom AutoML Task")
    st.markdown("Create a custom AutoML workflow with specific requirements.")
    
    with st.form("custom_task_form"):
        task_name = st.text_input("Task Name", placeholder="e.g., Customer Retention Prediction")
        task_description = st.text_area(
            "Task Description", 
            placeholder="Describe what you want to achieve with this ML task...",
            height=100
        )
        
        col1, col2 = st.columns(2)
        with col1:
            target_column = st.text_input("Target Column", placeholder="e.g., churn")
            task_type = st.selectbox(
                "Task Type", 
                ["classification", "regression", "clustering", "time_series", "nlp", "computer_vision"]
            )
        
        with col2:
            quality_threshold = st.slider("Quality Threshold", 0.5, 1.0, 0.8, 0.05)
            max_iterations = st.number_input("Max Iterations", 1, 10, 3)
        
        # Agent selection with descriptions
        st.subheader("Agent Selection")
        
        agent_descriptions = {
            "eda": "📊 Exploratory Data Analysis - Statistical analysis and visualizations",
            "data_hygiene": "🧹 Data Hygiene - Cleaning, missing values, outliers",
            "feature_engineering": "⚙️ Feature Engineering - Create and transform features", 
            "classification": "🎯 Classification - Binary/multiclass prediction models",
            "regression": "📈 Regression - Continuous value prediction models",
            "time_series": "📅 Time Series - Temporal data forecasting",
            "nlp": "💬 NLP - Text processing and natural language tasks",
            "computer_vision": "👁️ Computer Vision - Image classification and analysis",
            "clustering": "🎨 Clustering - Unsupervised grouping and segmentation",
            "ensemble": "🤝 Ensemble - Combine multiple models for better performance",
            "hyperparameter_tuning": "🔧 Hyperparameter Tuning - Optimize model parameters",
            "qa": "🔍 Quality Assurance - Validate workflow quality and consistency"
        }
        
        # Generate intelligent agent recommendations from task description
        def analyze_task_description(description, task_type):
            """Analyze task description to recommend appropriate agents."""
            if not description:
                # Fallback to task type if no description
                fallback_workflows = {
                    "classification": ["eda", "data_hygiene", "feature_engineering", "classification"],
                    "regression": ["eda", "data_hygiene", "feature_engineering", "regression"],
                    "clustering": ["eda", "data_hygiene", "feature_engineering", "clustering"],
                    "time_series": ["eda", "time_series"],
                    "nlp": ["eda", "nlp"],
                    "computer_vision": ["computer_vision"]
                }
                return fallback_workflows.get(task_type, ["eda", "classification"])
            
            description_lower = description.lower()
            recommended = []
            
            # Always start with EDA for data understanding (unless pure computer vision)
            if "image" not in description_lower and "photo" not in description_lower and "picture" not in description_lower:
                recommended.append("eda")
            
            # Data quality indicators
            if any(word in description_lower for word in ["missing", "dirty", "clean", "outlier", "quality", "incomplete"]):
                recommended.append("data_hygiene")
            
            # Feature engineering indicators
            if any(word in description_lower for word in ["feature", "transform", "encode", "scale", "normalize", "engineer"]):
                recommended.append("feature_engineering")
            
            # Task type detection from description (using separate if statements to allow multiple matches)
            
            # Regression indicators (check first for satisfaction/rating/score tasks)
            if any(word in description_lower for word in [
                "satisfaction", "rating", "score", "rate", "value", "price", "amount", "level",
                "regression", "continuous", "predict value", "estimate", "forecast value"
            ]):
                if "regression" not in recommended:
                    recommended.append("regression")
            
            # Classification indicators
            elif any(word in description_lower for word in [
                "classify", "classification", "category", "categorize", "class", "label", 
                "type", "group", "binary", "yes/no", "true/false"
            ]):
                if "classification" not in recommended:
                    recommended.append("classification")
            
            # Handle "predict" - more specific analysis needed
            elif "predict" in description_lower:
                # Look for context clues to determine if it's classification or regression
                if any(word in description_lower for word in [
                    "satisfaction", "rating", "score", "value", "amount", "price", "level", "degree"
                ]):
                    if "regression" not in recommended:
                        recommended.append("regression")
                else:
                    # Default predict to classification if no clear regression indicators
                    if "classification" not in recommended:
                        recommended.append("classification")
            
            # Time series indicators
            if any(word in description_lower for word in ["time series", "forecast", "temporal", "daily", "monthly", "trend", "seasonal"]):
                recommended.append("time_series")
            
            # Clustering indicators
            if any(word in description_lower for word in ["cluster", "group", "segment", "unsupervised", "similarity"]):
                recommended.append("clustering")
            
            # NLP indicators
            if any(word in description_lower for word in ["text", "nlp", "sentiment", "language", "review", "comment", "survey", "feedback"]):
                recommended.append("nlp")
            
            # Computer vision indicators
            if any(word in description_lower for word in ["image", "photo", "picture", "vision", "visual"]):
                recommended.append("computer_vision")
            
            # Auto-add standard pipeline components for ML tasks
            has_ml_task = any(agent in recommended for agent in ["classification", "regression", "clustering"])
            
            if has_ml_task:
                # Add data hygiene if not already present and not explicitly excluded
                if "data_hygiene" not in recommended:
                    if not any(word in description_lower for word in ["clean data", "already clean", "preprocessed"]):
                        recommended.insert(-1, "data_hygiene")  # Add before the ML task
                
                # Add feature engineering for standard ML pipelines
                if "feature_engineering" not in recommended:
                    if any(word in description_lower for word in ["pipeline", "feature", "transform", "preprocess"]) or has_ml_task:
                        # Insert feature engineering before the ML task
                        ml_task_idx = next((i for i, agent in enumerate(recommended) if agent in ["classification", "regression", "clustering"]), len(recommended))
                        recommended.insert(ml_task_idx, "feature_engineering")
            
            # Performance optimization indicators
            if any(word in description_lower for word in ["optimize", "best", "high accuracy", "performance", "tune"]):
                if any(agent in recommended for agent in ["classification", "regression"]) and "hyperparameter_tuning" not in recommended:
                    recommended.append("hyperparameter_tuning")
            
            # Ensemble indicators  
            if any(word in description_lower for word in ["ensemble", "combine models", "voting", "stacking"]):
                recommended.append("ensemble")
            
            # Default fallback - ensure we have at least EDA + one ML task
            if not recommended:
                recommended = ["eda", "classification"]
            elif len(recommended) == 1 and recommended[0] == "eda":
                # If only EDA, add a default ML task based on task_type
                if task_type == "regression":
                    recommended.append("regression")
                elif task_type == "clustering":
                    recommended.append("clustering")
                else:
                    recommended.append("classification")
            
            return recommended
        
        # Generate recommendations based on task description
        suggested = analyze_task_description(task_description, task_type)
        
        # Show AI-generated recommendation with reasoning
        if task_description:
            st.success(f"🤖 AI Recommendation based on your description: **{' → '.join(suggested)}**")
            with st.expander("Why these agents?"):
                reasoning = []
                if "eda" in suggested:
                    reasoning.append("📊 **EDA**: Understand data patterns and characteristics")
                if "data_hygiene" in suggested:
                    if any(word in task_description.lower() for word in ["missing", "dirty", "clean", "outlier", "quality"]):
                        reasoning.append("🧹 **Data Hygiene**: Your description mentions data quality concerns")
                    else:
                        reasoning.append("🧹 **Data Hygiene**: Standard data cleaning for ML pipeline")
                        
                if "feature_engineering" in suggested:
                    if any(word in task_description.lower() for word in ["feature", "transform", "encode", "scale"]):
                        reasoning.append("⚙️ **Feature Engineering**: Your description mentions feature work")
                    else:
                        reasoning.append("⚙️ **Feature Engineering**: Prepare features for ML models")
                        
                if "classification" in suggested:
                    reasoning.append("🎯 **Classification**: Task involves predicting categories/classes")
                    
                if "regression" in suggested:
                    # Be more specific about why regression was chosen
                    regression_keywords = ["satisfaction", "rating", "score", "value", "amount", "level"]
                    detected_keywords = [word for word in regression_keywords if word in task_description.lower()]
                    if detected_keywords:
                        reasoning.append(f"📈 **Regression**: Detected continuous prediction task ('{', '.join(detected_keywords)}')")
                    else:
                        reasoning.append("📈 **Regression**: Task involves predicting continuous values")
                if "time_series" in suggested:
                    reasoning.append("📅 **Time Series**: Detected temporal/forecasting requirements")
                if "nlp" in suggested:
                    reasoning.append("💬 **NLP**: Text processing task identified")
                if "hyperparameter_tuning" in suggested:
                    reasoning.append("🔧 **Hyperparameter Tuning**: Performance optimization requested")
                
                for reason in reasoning:
                    st.markdown(reason)
        else:
            st.info(f"💡 Default workflow for {task_type}: {' → '.join(suggested)}")
        
        agents = st.multiselect(
            "Select Agents for Your Workflow (customize the AI recommendation)",
            options=list(agent_descriptions.keys()),
            default=suggested,
            format_func=lambda x: agent_descriptions[x]
        )
        
        uploaded_file = st.file_uploader("Upload Dataset", type=['csv'])
        
        submitted = st.form_submit_button("Run Custom AutoML Task")
        
        if submitted:
            if uploaded_file and task_description and target_column:
                st.success(f"Custom task '{task_name}' submitted!")
                st.info("This would execute a custom AutoML workflow with your specifications.")
                
                with st.expander("Task Configuration"):
                    st.json({
                        "task_name": task_name,
                        "task_description": task_description,
                        "target_column": target_column,
                        "task_type": task_type,
                        "quality_threshold": quality_threshold,
                        "max_iterations": max_iterations,
                        "selected_agents": agents,
                        "dataset": uploaded_file.name if uploaded_file else None
                    })
            else:
                st.error("Please fill in all required fields and upload a dataset.")

def show_agent_analytics():
    """Show analytics and performance metrics for agents."""
    st.header("📊 Agent Analytics & Performance")
    st.markdown("Monitor agent performance and system metrics.")
    
    # Simulated metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Agents", "8", "+2")
    with col2:
        st.metric("Tasks Completed", "1,247", "+156")
    with col3:
        st.metric("Average Accuracy", "87.3%", "+2.1%")
    with col4:
        st.metric("Success Rate", "94.2%", "+0.8%")
    
    # Agent performance chart
    st.subheader("Agent Performance Over Time")
    
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    agents = ['EDA Agent', 'Classification Agent', 'Feature Engineering', 'Data Hygiene']
    
    performance_data = []
    for agent in agents:
        for date in dates:
            performance_data.append({
                'Date': date,
                'Agent': agent,
                'Success Rate': np.random.normal(0.9, 0.05),
                'Avg Processing Time': np.random.normal(45, 10)
            })
    
    df_perf = pd.DataFrame(performance_data)
    
    # Success rate chart
    import plotly.express as px
    fig1 = px.line(df_perf, x='Date', y='Success Rate', color='Agent', 
                   title='Agent Success Rates Over Time')
    fig1.update_yaxes(range=[0.8, 1.0])
    st.plotly_chart(fig1, use_container_width=True)
    
    # Processing time chart
    fig2 = px.line(df_perf, x='Date', y='Avg Processing Time', color='Agent',
                   title='Average Processing Time (seconds)')
    st.plotly_chart(fig2, use_container_width=True)
    
    # Agent status table
    st.subheader("Current Agent Status")
    status_data = {
        'Agent': agents,
        'Status': ['Active', 'Active', 'Active', 'Idle'],
        'Current Tasks': [3, 2, 1, 0],
        'Queue Length': [5, 3, 2, 0],
        'Last Activity': ['2 min ago', '1 min ago', '5 min ago', '1 hour ago']
    }
    st.dataframe(pd.DataFrame(status_data), use_container_width=True)

def show_agent_communication():
    """Show agent communication and collaboration patterns."""
    st.header("🔬 Agent Communication Hub")
    st.markdown("Monitor inter-agent communication and collaboration patterns.")
    
    # Communication network
    st.subheader("Agent Communication Network")
    st.info("This shows how agents communicate and share information during workflows.")
    
    # Simulated communication log
    st.subheader("Recent Agent Communications")
    
    communication_log = [
        {"Time": "10:32:15", "From": "EDA Agent", "To": "Feature Engineering", "Message": "Data profile complete, found 3 categorical features"},
        {"Time": "10:32:18", "From": "Feature Engineering", "To": "Classification Agent", "Message": "Features encoded, ready for training"},
        {"Time": "10:32:45", "From": "Classification Agent", "To": "EDA Agent", "Message": "Model accuracy: 89.2%, requesting feature importance"},
        {"Time": "10:33:02", "From": "Data Hygiene", "To": "All Agents", "Message": "Data quality check complete, no issues found"},
        {"Time": "10:33:15", "From": "Classification Agent", "To": "Router Agent", "Message": "Classification task completed successfully"}
    ]
    
    for log in communication_log:
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 4])
            with col1:
                st.code(log["Time"])
            with col2:
                st.markdown(f"**{log['From']}** → **{log['To']}**")
            with col3:
                st.markdown(log["Message"])
            st.divider()
    
    # Communication statistics
    st.subheader("Communication Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Messages Today", "1,423", "+234")
    with col2:
        st.metric("Active Conversations", "12", "+3")
    with col3:
        st.metric("Avg Response Time", "0.8s", "-0.2s")
    
    # Message types chart
    st.subheader("Message Types Distribution")
    
    message_types = {
        'Data Transfer': 35,
        'Status Updates': 28,
        'Error Reports': 8,
        'Task Requests': 22,
        'Results Sharing': 7
    }
    
    import plotly.express as px
    fig = px.pie(values=list(message_types.values()), 
                names=list(message_types.keys()),
                title="Distribution of Inter-Agent Message Types")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
