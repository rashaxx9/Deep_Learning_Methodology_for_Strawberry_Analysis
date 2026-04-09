import streamlit as st
import numpy as np
import random
from PIL import Image
import time


def mock_predict():
    """
    Simulates AI prediction without using PyTorch.
    Returns: (prediction, confidence, all_probabilities)
    """
 
    base = random.random()
    
    if base < 0.6:  # 60% chance of ripe
        probabilities = [0.15, 0.75, 0.10]
        prediction = "Ripe"
    elif base < 0.8:  # 20% chance of unripe
        probabilities = [0.70, 0.20, 0.10]
        prediction = "Unripe"
    else:  # 20% chance of overripe
        probabilities = [0.10, 0.20, 0.70]
        prediction = "Overripe"
    
    confidence = max(probabilities) * 100
    

    probabilities = [p + random.uniform(-0.05, 0.05) for p in probabilities]
    probabilities = [max(0, min(1, p)) for p in probabilities]
    total = sum(probabilities)
    probabilities = [p / total for p in probabilities]
    
    return prediction, confidence, probabilities



def preprocess_image(image):
    """Resize image to expected format without torch"""
    image = image.resize((128, 128))
    return image



class_names = ["Overripe", "Ripe", "Unripe"]
class_emojis = ["🔴", "🍓", "🟢"]
class_colors = ["#856404", "#155724", "#721c24"]
class_bgs = ["#fff3cd", "#d4edda", "#f8d7da"]
class_messages = [
    "⚠️ Oh no! This strawberry is overripe. Use it today! 🥺",
    "🍓 YUM! This strawberry is perfectly ripe! Enjoy! 😋",
    "⏳ This strawberry needs more time to ripen. Be patient! 🌱"
]

# Map prediction names to indices
prediction_to_index = {"Overripe": 0, "Ripe": 1, "Unripe": 2}


st.set_page_config(
    page_title="Strawberry Ripeness Classifier 🍓",
    page_icon="🍓",
    layout="centered"
)


st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #ffe4e1 0%, #ffc0cb 100%);
    }
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        color: #8b0000;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #8b0000;
    }
    .result-box {
        text-align: center;
        padding: 20px;
        border-radius: 20px;
        margin-top: 20px;
    }
    .stButton button {
        background-color: #ff4b4b;
        color: white;
        font-size: 1.2rem;
        border-radius: 30px;
        padding: 10px 30px;
    }
    .stButton button:hover {
        background-color: #8b0000;
    }
</style>
""", unsafe_allow_html=True)


st.markdown('<p class="main-title">🍓 Strawberry Ripeness Classifier 🍓</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a strawberry photo and I\'ll tell you if it\'s ready to eat!</p>', unsafe_allow_html=True)

st.divider()


uploaded_file = st.file_uploader(
    "Choose a strawberry image...",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear photo of a strawberry"
)


if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Your Strawberry 🍓", width=300)
    
  
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze Strawberry", type="primary", use_container_width=True)
    
    if analyze_button:
        with st.spinner("🍓 Analyzing your strawberry... 🍓"):
            # Simulate processing time (like real AI)
            time.sleep(1.5)
  
            prediction, confidence, probabilities = mock_predict()
            predicted_class = prediction_to_index[prediction]
        
        st.divider()
       
        emoji = class_emojis[predicted_class]
        result_text = class_names[predicted_class].upper()
        
        st.markdown(f"""
        <div class="result-box" style="background-color: {class_bgs[predicted_class]};">
            <h1 style="font-size: 4rem;">{emoji}</h1>
            <h2 style="color: {class_colors[predicted_class]};">{result_text}</h2>
            <p style="font-size: 1.2rem;">{class_messages[predicted_class]}</p>
        </div>
        """, unsafe_allow_html=True)

        st.write(f"**Confidence:** {confidence:.1f}%")
        st.progress(confidence / 100)
        
   
        with st.expander("📊 See detailed analysis"):
            for i, name in enumerate(class_names):
                prob = probabilities[i] * 100
                st.write(f"{class_emojis[i]} {name}: {prob:.1f}%")
                st.progress(prob / 100)
        
       
        st.caption("🤖 AI Model: Mock Predictions (PyTorch model will be added when Streamlit Cloud supports Python 3.14)")

st.divider()
st.caption("🍓 Made with love for strawberries 🍓")
st.caption("⚠️ For educational purposes only")
