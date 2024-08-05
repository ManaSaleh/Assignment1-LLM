import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from sentence_transformers.util import cos_sim


class ImageSearcher:
    def __init__(self, model_name="openai/clip-vit-base-patch32", image_dir="Script/images"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.image_dir = image_dir

    def load_images(self):
        image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_tensors = [Image.open(path).convert("RGB") for path in image_paths[:1000]]
        return image_paths, image_tensors

    def search_images(self, search_text, image_tensors):
        inputs = self.processor(
            text=[search_text],
            images=image_tensors,
            return_tensors='pt', padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds

            similarity_scores = cos_sim(image_features, text_features)

        max_score = similarity_scores.max().item()
        best_image_index = similarity_scores.argmax().item()
        return best_image_index, max_score
