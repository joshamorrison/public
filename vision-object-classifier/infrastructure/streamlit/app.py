"""
Vision Object Classifier - Streamlit Web App
Interactive web interface for dish cleanliness classification
"""

import streamlit as st
import requests
from PIL import Image
import io
import sys
from pathlib import Path
import time
import json

# Add src to path for local imports
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

# Configure Streamlit page
st.set_page_config(
    page_title="Vision Object Classifier",
    page_icon="🍽️",
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
    .prediction-clean {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .prediction-dirty {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_demo_images():
    """Load demo images for quick testing"""
    demo_images = {}
    samples_dir = project_root / "data" / "samples" / "demo_images"
    
    if samples_dir.exists():
        for img_path in samples_dir.glob("*.jpg"):
            demo_images[img_path.name] = str(img_path)
    
    return demo_images

def classify_with_api(image_file, api_url, model_type="balanced", return_confidence=True):
    """Classify image using the FastAPI endpoint"""
    try:
        files = {"image": image_file}
        data = {
            "model_type": model_type,
            "return_confidence": return_confidence,
            "min_confidence": 0.0
        }
        
        response = requests.post(f"{api_url}/api/v1/classify/single", files=files, data=data)
        response.raise_for_status()
        
        return response.json()
    
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None

def classify_with_local_model(image):
    """Classify image using local model (fallback when API is not available)"""
    try:
        from vision_classifier.predict import DishCleanlinessPredictor
        
        # Find available model
        models_dir = project_root / "models"
        for model_name in ["final_balanced_model.pth", "balanced_model.pth", "fast_model.pth"]:
            model_path = models_dir / model_name
            if model_path.exists():
                break
        else:
            st.error("No trained models found")
            return None
        
        # Initialize predictor
        if 'predictor' not in st.session_state:
            with st.spinner("Loading model..."):
                st.session_state.predictor = DishCleanlinessPredictor(
                    model_path=str(model_path),
                    config_path=None
                )
        
        # Make prediction
        predictor = st.session_state.predictor
        
        # Save image temporarily
        temp_path = "/tmp/temp_image.jpg"
        image.save(temp_path)
        
        result = predictor.predict_single(temp_path)
        
        # Convert to API-like format
        predicted_class = "clean" if result['prediction'] == 0 else "dirty"
        
        return {
            "success": True,
            "result": {
                "predicted_class": predicted_class,
                "confidence": result['confidence'],
                "probabilities": {
                    "clean": result['clean_prob'],
                    "dirty": result['dirty_prob']
                }
            },
            "model_type": "local"
        }
        
    except Exception as e:
        st.error(f"Local classification failed: {str(e)}")
        return None

def display_prediction_result(result, image):
    """Display prediction results with styling"""
    if not result or not result.get("success"):
        st.error("Classification failed")
        return
    
    pred_result = result["result"]
    predicted_class = pred_result["predicted_class"]
    confidence = pred_result.get("confidence", 0)
    probabilities = pred_result.get("probabilities", {})
    
    # Display image and prediction side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Image")
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.subheader("Classification Result")
        
        # Prediction with color coding
        if predicted_class.lower() == "clean":
            st.markdown(f"""
            <div class="prediction-clean">
                <h3>✨ CLEAN DISH</h3>
                <p>The dish appears to be clean and ready for use.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-dirty">
                <h3>🍝 DIRTY DISH</h3>
                <p>The dish appears to have food residue or stains.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Confidence display
        if confidence:
            confidence_pct = confidence * 100
            if confidence_pct >= 80:
                confidence_class = "confidence-high"
                confidence_text = "High"
            elif confidence_pct >= 60:
                confidence_class = "confidence-medium" 
                confidence_text = "Medium"
            else:
                confidence_class = "confidence-low"
                confidence_text = "Low"
            
            st.markdown(f"""
            <p><strong>Confidence:</strong> 
            <span class="{confidence_class}">{confidence_pct:.1f}% ({confidence_text})</span></p>
            """, unsafe_allow_html=True)
        
        # Probability breakdown
        if probabilities:
            st.subheader("Probability Breakdown")
            clean_prob = probabilities.get("clean", 0) * 100
            dirty_prob = probabilities.get("dirty", 0) * 100
            
            st.progress(clean_prob / 100)
            st.text(f"Clean: {clean_prob:.1f}%")
            
            st.progress(dirty_prob / 100)  
            st.text(f"Dirty: {dirty_prob:.1f}%")
        
        # Model info
        model_type = result.get("model_type", "unknown")
        st.text(f"Model: {model_type}")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🍽️ Vision Object Classifier</h1>', unsafe_allow_html=True)
    st.markdown("**AI-powered clean/dirty dish classification**")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # API URL configuration
    use_api = st.sidebar.checkbox("Use API", value=False, help="Use FastAPI backend instead of local model")
    
    if use_api:
        api_url = st.sidebar.text_input("API URL", value="http://localhost:8000", help="FastAPI server URL")
        model_type = st.sidebar.selectbox("Model Type", ["fast", "balanced", "accurate"], index=1)
    else:
        st.sidebar.info("Using local model inference")
        api_url = None
        model_type = "local"
    
    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.0, 0.1)
        return_probabilities = st.checkbox("Show Probabilities", value=True)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📷 Upload Image", "🖼️ Demo Images", "📊 Batch Processing"])
    
    with tab1:
        st.header("Upload Your Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            
            # Classify button
            if st.button("Classify Image", type="primary"):
                with st.spinner("Classifying..."):
                    if use_api and api_url:
                        # Use API
                        uploaded_file.seek(0)  # Reset file pointer
                        result = classify_with_api(uploaded_file, api_url, model_type, return_probabilities)
                    else:
                        # Use local model
                        result = classify_with_local_model(image)
                    
                    if result:
                        display_prediction_result(result, image)
    
    with tab2:
        st.header("Demo Images")
        demo_images = load_demo_images()
        
        if demo_images:
            selected_demo = st.selectbox("Choose a demo image:", list(demo_images.keys()))
            
            if selected_demo:
                demo_image_path = demo_images[selected_demo]
                demo_image = Image.open(demo_image_path)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(demo_image, caption=selected_demo, use_column_width=True)
                
                with col2:
                    if st.button("Classify Demo Image", type="primary"):
                        with st.spinner("Classifying..."):
                            if use_api and api_url:
                                # Convert PIL image to file-like object
                                img_buffer = io.BytesIO()
                                demo_image.save(img_buffer, format='JPEG')
                                img_buffer.seek(0)
                                result = classify_with_api(img_buffer, api_url, model_type, return_probabilities)
                            else:
                                result = classify_with_local_model(demo_image)
                            
                            if result:
                                st.success("Classification complete!")
                                display_prediction_result(result, demo_image)
        else:
            st.info("No demo images available. Upload your own images in the Upload tab.")
    
    with tab3:
        st.header("Batch Processing")
        st.info("Batch processing feature - Upload multiple images at once")
        
        uploaded_files = st.file_uploader("Choose multiple images...", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
        
        if uploaded_files and len(uploaded_files) > 0:
            st.write(f"Selected {len(uploaded_files)} images")
            
            if st.button("Process All Images", type="primary"):
                results = []
                progress_bar = st.progress(0)
                
                for i, uploaded_file in enumerate(uploaded_files):
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    image = Image.open(uploaded_file)
                    
                    if use_api and api_url:
                        uploaded_file.seek(0)
                        result = classify_with_api(uploaded_file, api_url, model_type, return_probabilities)
                    else:
                        result = classify_with_local_model(image)
                    
                    if result and result.get("success"):
                        pred_result = result["result"]
                        results.append({
                            "filename": uploaded_file.name,
                            "prediction": pred_result["predicted_class"],
                            "confidence": pred_result.get("confidence", 0) * 100
                        })
                    else:
                        results.append({
                            "filename": uploaded_file.name,
                            "prediction": "Error",
                            "confidence": 0
                        })
                
                # Display results
                st.subheader("Batch Results")
                
                # Summary
                clean_count = sum(1 for r in results if r["prediction"] == "clean")
                dirty_count = sum(1 for r in results if r["prediction"] == "dirty")
                error_count = sum(1 for r in results if r["prediction"] == "Error")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Clean Dishes", clean_count)
                col2.metric("Dirty Dishes", dirty_count)  
                col3.metric("Errors", error_count)
                
                # Detailed results
                st.dataframe(results, use_container_width=True)
                
                # Download results
                csv = "\n".join([",".join(map(str, r.values())) for r in results])
                csv_header = "filename,prediction,confidence\n" + csv
                st.download_button(
                    label="Download Results as CSV",
                    data=csv_header,
                    file_name="classification_results.csv",
                    mime="text/csv"
                )
    
    # Footer
    st.markdown("---")
    st.markdown("**Vision Object Classifier** - Built with Streamlit and FastAPI")

if __name__ == "__main__":
    main()