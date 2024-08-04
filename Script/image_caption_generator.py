from PIL import Image
from transformers import pipeline

class ImageCaptionGenerator:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        self.pipe = pipeline("image-to-text", model=model_name)

    def generate_caption(self, image):
        if image.mode != 'RGB':
            image = image.convert('RGB')        
        caption = self.pipe(image)[0]['generated_text']
        return caption
