import streamlit as st
from transformers import pipeline
from PIL import Image

# Title of the Streamlit app
st.title('Streamlit Drag and Drop Image Classification App')

# File uploader with drag and drop
uploaded_file = st.file_uploader("Drag and drop an image file here or click to upload", type=["jpg", "jpeg", "png"])

# Check if a file has been uploaded
if uploaded_file is not None:
    try:
        # Open and display the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        # Load the image classification pipeline
        classifier = pipeline("image-classification", model="microsoft/resnet-50")
        
        # Perform classification
        classification_message = st.empty()
        classification_message.write("Classifying the image...")
        results = classifier(image)
        classification_message.empty()
        
        # Display results as text with styled bars
        st.markdown("### Classification Results with Styled Bars:")
        for result in results:
            st.write(f"**Label:** {result['label']} ({result['score'] * 100:.2f}%)")
            st.progress(int(result['score'] * 100))
        
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.write("Drag and drop an image file to get started.")
