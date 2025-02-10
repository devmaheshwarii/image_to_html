import streamlit as st
import torch
from PIL import Image
from pathlib import Path
from model import ImageToHTML
from utils import setup_logging, load_checkpoint
from config import ModelConfig
import clip
from transformers import GPT2LMHeadModel, GPT2Tokenizer

logger = setup_logging(__name__)
config = ModelConfig()

def load_models():
    """Load all required models"""
    try:
        # Load CLIP
        clip_model, clip_preprocess = clip.load(config.clip_model_name, device=config.device)
        
        # Load GPT2
        gpt2_model = GPT2LMHeadModel.from_pretrained(config.gpt2_model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(config.gpt2_model_name)
        
        # Initialize our model
        model = ImageToHTML(gpt2_model, config.clip_dim)
        
        # Load latest checkpoint if exists
        checkpoints = list(config.checkpoint_dir.glob("*.pt"))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
            load_checkpoint(str(latest_checkpoint), model)
            logger.info(f"Loaded checkpoint: {latest_checkpoint}")
        
        return model, clip_model, clip_preprocess, tokenizer
    
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

def main():
    st.title("Image to HTML Converter")
    st.write("Upload an image to generate its HTML code")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        try:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Load models
            model, clip_model, clip_preprocess, tokenizer = load_models()
            
            if st.button('Generate HTML'):
                with st.spinner('Generating HTML...'):
                    # Process image
                    image = clip_preprocess(image).unsqueeze(0).to(config.device)
                    
                    with torch.no_grad():
                        # Generate HTML
                        image_features = clip_model.encode_image(image)
                        generated_ids = model.generate(image_features)
                        generated_html = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    
                    # Display results
                    st.code(generated_html, language='html')
                    
                    # Add download button
                    st.download_button(
                        label="Download HTML",
                        data=generated_html,
                        file_name="generated.html",
                        mime="text/html"
                    )
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()