import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

class StrawberryCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(StrawberryCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


@st.cache_resource
def load_model():
    model = StrawberryCNN(num_classes=3)
  
    possible_paths = [
    "experiments\baseline\best_model.pth",
]
    
    for path in possible_paths:
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            return model, path
    
    st.error("No model found! Please train a model first.")
    return None, None



transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



class_names = ["Overripe", "Ripe", "Unripe"]
class_emojis = ["🔴", "🍓", "🟢"]
class_colors = ["#856404", "#155724", "#721c24"]
class_bgs = ["#fff3cd", "#d4edda", "#f8d7da"]
class_messages = [
    "⚠️ Oh no! This strawberry is overripe. Use it today! 🥺",
    "🍓 YUM! This strawberry is perfectly ripe! Enjoy! 😋",
    "⏳ This strawberry needs more time to ripen. Be patient! 🌱"
]



st.set_page_config(
    page_title="Strawberry Ripeness Classifier 🍓",
    page_icon="🍓",
    layout="centered"
)

# Custom CSS for cute styling
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

# Title
st.markdown('<p class="main-title">🍓 Strawberry Ripeness Classifier 🍓</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a strawberry photo and I\'ll tell you if it\'s ready to eat!</p>', unsafe_allow_html=True)

st.divider()

# File uploader
uploaded_file = st.file_uploader(
    "Choose a strawberry image...",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear photo of a strawberry"
)

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Your Strawberry 🍓", width=300)
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze Strawberry", type="primary", use_container_width=True)
    
    if analyze_button:
        with st.spinner("🍓 Analyzing your strawberry... 🍓"):
            # Preprocess
            input_tensor = transform(image).unsqueeze(0)
            
            # Predict
            model, model_path = load_model()
            if model is not None:
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item() * 100
        
        if model is not None:
            st.divider()
            
            # Result box
            emoji = class_emojis[predicted_class]
            result_text = class_names[predicted_class].upper()
            
            st.markdown(f"""
            <div class="result-box" style="background-color: {class_bgs[predicted_class]};">
                <h1 style="font-size: 4rem;">{emoji}</h1>
                <h2 style="color: {class_colors[predicted_class]};">{result_text}</h2>
                <p style="font-size: 1.2rem;">{class_messages[predicted_class]}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence bar
            st.write(f"**Confidence:** {confidence:.1f}%")
            st.progress(confidence / 100)
            
            # Show all probabilities
            with st.expander("📊 See detailed analysis"):
                for i, name in enumerate(class_names):
                    prob = probabilities[0][i].item() * 100
                    st.write(f"{class_emojis[i]} {name}: {prob:.1f}%")
                    st.progress(prob / 100)
            
            st.caption(f"🤖 Model loaded from: {model_path}")
        else:
            st.error("No trained model found. Please train a model first!")

st.divider()
st.caption("🍓 Made with love for strawberries 🍓")
st.caption("⚠️ For educational purposes only")
