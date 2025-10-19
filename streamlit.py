import streamlit as st
import requests
from PIL import Image
import os
import tempfile

API_URL = "http://127.0.0.1:8089/unikrew/inference"

st.set_page_config(page_title="Intelligent Receipt OCR", layout="wide")

col1, col2 = st.columns([1, 1])

with col1:
    st.title("🧾 Intelligent Receipt OCR")

    uploaded_file = st.file_uploader("Upload a receipt image", type=["jpg", "jpeg", "png"])
    result = None
    temp_path = None

    if uploaded_file is not None:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = os.path.abspath(temp_file.name)

    if st.button("Run Inference"):
        if uploaded_file is None:
            st.warning("Please upload an image first.")
        elif not temp_path or not os.path.exists(temp_path):
            st.error("Temporary image file not found.")
        else:
            with st.spinner("Scanning..."):
                try:
                    response = requests.post(API_URL, json={"image_path": temp_path})
                    if response.status_code == 200:
                        result = response.json()
                    else:
                        st.error(f"Server Error: {response.status_code}")
                        st.text(response.text)
                except Exception as e:
                    st.error(f"Error connecting to API: {e}")

    if result:
        st.success("Inference complete!")
        st.json(result)

with col2:
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Selected Image (Preview)", width=200)