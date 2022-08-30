import streamlit as st
from PIL import Image

from utils import Predict

# Page settings
st.set_page_config(
    page_title="Image Classification",
    layout="wide",
    initial_sidebar_state="expanded"
 )

# Title
st.title('Classification of Chest X-ray Images')

# Upload file
uploaded_file = st.file_uploader(label="Choose a file", type=['jpg', 'jpeg','png'])

sidebar = st.sidebar

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    # image = np.array(image)
    predict = Predict()

    col1, col2 = st.columns([0.5, 0.5])

    #Col 1
    with col1:
        st.markdown('<p style="text-align: center;">Input Image</p>', unsafe_allow_html=True)
        st.image(image, width=425, caption="X-Ray Image")

    #Col 2
    with col2:
        pred,topk = predict.predict(image)
        st.text("Prediction: "+ pred)
        st.text("Confidence: "+ str(topk))