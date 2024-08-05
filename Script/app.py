import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from sentence_transformers.util import cos_sim
import streamlit as st
from image_caption_generator import ImageCaptionGenerator
from image_searcher import ImageSearcher

caption_generator = ImageCaptionGenerator()
image_searcher = ImageSearcher()

def search_images_by_caption():
    st.header("Search for Images by Caption")
    search_text = st.text_input("Enter a caption to search for images")

    if search_text:
        if not os.path.exists(image_searcher.image_dir):
            st.error(f"Directory '{image_searcher.image_dir}' does not exist.")
        else:
            image_paths, image_tensors = image_searcher.load_images()

            if not image_paths:
                st.error("No images found in the specified directory.")
            else:
                try:
                    best_image_index, max_score = image_searcher.search_images(search_text, image_tensors)
                    best_image = Image.open(image_paths[best_image_index])
                    st.image(best_image, caption=f"Best Match\nCaption: {search_text}\nSimilarity Score: {max_score:.2f}")
                except Exception as e:
                    st.error(f"An error occurred while searching for images: {e}")

def upload_image_and_generate_caption():
    st.header("Image Upload and Caption Generation")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            with st.spinner("Generating caption..."):
                caption = caption_generator.generate_caption(image)
                caption_with_title = caption.capitalize()
                st.markdown(f'#### {caption_with_title}')
        except Exception as e:
            st.error(f"An error occurred while processing the image: {e}")
    else:
        st.write("Please upload an image file.")

def main():
    st.title("Image Search and Caption Generation")
    search_images_by_caption()
    upload_image_and_generate_caption()

if __name__ == "__main__":
    main()
