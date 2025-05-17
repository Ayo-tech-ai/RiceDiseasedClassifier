import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import gdown
import os

# -------------------------------
# 1. Load Model from Google Drive
# -------------------------------

MODEL_PATH = "RiceClassifier.pth"
FILE_ID = "13nlieOIczZPmbCaA8M2AlefOrXINTXyL"  # Your actual Drive file ID

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    # Force loading full model (architecture + weights)
    model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model.eval()
    return model

model = load_model()

# -------------------------------
# 2. Class Names (Adjust if needed)
# -------------------------------

class_names = [
    'bacterial_leaf_blight', 'bacterial_leaf_streak', 'bakanae',
    'brown_spot', 'grassy_stunt_virus', 'healthy_rice_plant',
    'narrow_brown_spot', 'ragged_stunt_virus', 'rice_blast',
    'rice_false_smut', 'sheath_blight', 'sheath_rot',
    'stem_rot', 'tungro_virus'
]

# -------------------------------
# 3. Image Preprocessing Function
# -------------------------------

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image = image.convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

# -------------------------------
# 4. Streamlit Interface
# -------------------------------

st.title("Rice Leaf Disease Classifier")
st.write("Upload an image of a rice leaf and get a prediction of the disease class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            input_tensor = preprocess_image(image)
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probs, 1)

            label = class_names[predicted_class.item()]
            confidence_score = confidence.item() * 100

            st.success(f"**Prediction:** {label}")
            st.info(f"**Confidence Level:** {confidence_score:.2f}%")
