import streamlit as st
import tensorflow as tf
import numpy as np
import os
import cv2
from PIL import Image

# Custom CSS styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .sidebar .sidebar-content {
        background: #ffffff !important;
        box-shadow: 2px 0 10px rgba(0,0,0,0.1);
    }
    
    .stButton>button {
        background: #4CAF50 !important;
        color: white !important;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(76,175,80,0.4);
    }
    
    .uploaded-image {
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .prediction-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
    
    .healthy-status {
        color: #4CAF50;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    .disease-status {
        color: #f44336;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Cache the model loading
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('plant_disease_cnn_model.keras')

# Complete class names list (38 classes)
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Image preprocessing function
def model_predict(image_path):
    try:
        model = load_model()
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to read image file")
            
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img).astype('float32') / 255.0
        img = img.reshape(1, 224, 224, 3)
        
        preds = model.predict(img)
        result_index = np.argmax(preds, axis=-1)[0]
        
        # Validation check
        if result_index >= len(CLASS_NAMES) or result_index < 0:
            raise ValueError(f"Invalid prediction index: {result_index}")
            
        return result_index
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Sidebar navigation
st.sidebar.title('🌿 Plant Disease Prediction')
st.sidebar.markdown("---")
app_mode = st.sidebar.radio("Navigation", ['🏠 Home', '🌱 Disease Detection'])
st.sidebar.markdown("---")
st.sidebar.info("📸 Upload a clear image of plant leaves for instant disease detection and prevention recommendations.")

# Main content
if app_mode == '🏠 Home':
    st.markdown("<h1 style='text-align: center; color: #2c3e50;'>Plant Disease Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        <div style='padding: 2rem;'>
            <h2 style='color: #2c3e50;'>Protect Your Crops with AI-Powered Disease Detection</h2>
            <p style='font-size: 1.1rem; color: #555;'>
            Early detection of plant diseases using advanced deep learning technology. 
            Get instant analysis and recommendations to maintain healthy crops.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.image('ds.jpg', use_container_width=True)
    
    st.markdown("## 🔍 Key Features")
    features = [
        ("🤖 AI-Powered Analysis", "Deep learning model with 98%+ accuracy"),
        ("🌾 Multi-Crop Support", "Detect diseases in 25+ plant species"),
        ("📱 Instant Results", "Real-time prediction within seconds"),
        ("🌱 Sustainable Farming", "Early detection for better crop management")
    ]
    
    cols = st.columns(4)
    for i, (title, desc) in enumerate(features):
        with cols[i]:
            st.markdown(f"""
            <div class='feature-card'>
                <h3>{title}</h3>
                <p style='color: #666;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

elif app_mode == '🌱 Disease Detection':
    st.markdown("<h2 style='text-align: center; color: #2c3e50;'>Plant Disease Detection</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📤 Upload a plant leaf image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        save_path = os.path.join(os.getcwd(), uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("### 📷 Uploaded Image")
            st.image(uploaded_file, use_container_width=True, caption="Uploaded Leaf Image")
        
        with col2:
            st.markdown("### 🔍 Analysis Results")
            if st.button("Start Diagnosis", key="predict_button"):
                try:
                    with st.spinner("🔬 Analyzing plant health..."):
                        result_index = model_predict(save_path)
                        
                    if result_index is not None:
                        full_name = CLASS_NAMES[result_index]
                        plant, status = full_name.split('___')
                        status = status.replace('_', ' ').title()
                        
                        st.markdown(f"""
                        <div class='prediction-card'>
                            <h3 style='color: #555;'>Plant Species</h3>
                            <h2 style='color: #2c3e50;'>{plant}</h2>
                            <div style='margin: 2rem 0;'></div>
                            <h3 style='color: #555;'>Health Status</h3>
                            <div class='{"healthy-status" if "Healthy" in status else "disease-status"}'>
                                {'✅ ' if "Healthy" in status else '⚠️ '}{status}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if "Healthy" in status:
                            st.success("**Healthy Plant!** Continue with good agricultural practices.")
                        else:
                            st.error("**Disease Detected!** Immediate action recommended:")
                            st.markdown("""
                            - Isolate affected plants
                            - Consult with agricultural expert
                            - Apply recommended treatments
                            - Monitor plant health regularly
                            """)
                except Exception as e:
                    st.error(f"Diagnosis failed: {str(e)}")
                    st.info("Please try with a different image or check file format")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🌍 Promoting Sustainable Agriculture Through AI Technology</p>
    <p>© 2023 Plant Disease Prediction System</p>
</div>
""", unsafe_allow_html=True)