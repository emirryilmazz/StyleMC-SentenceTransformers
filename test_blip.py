import torch
from PIL import Image
from lavis.models import model_zoo, load_model_and_preprocess
from lavis.processors import load_processor
import requests

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(model_zoo)
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
model, vis_processors, text_processors = load_model_and_preprocess("blip2_feature_extractor", "pretrain_vitL", device=device, is_eval=True)
caption = "two brown cats and two remote controls in the pink couch"
text_input = text_processors["eval"](caption)

image = vis_processors["eval"](image).unsqueeze(0).to(device)
sample = {"image": image, "text_input": [text_input]}

features_image = model.extract_features(sample, mode="image")
features_text = model.extract_features(sample, mode="text")

# low-dimensional projected features
print(features_image.image_embeds_proj.shape)
# torch.Size([1, 197, 256])
print(features_text.text_embeds_proj.shape)
# torch.Size([1, 12, 256])
similarity = features_image.image_embeds_proj[:,0,:] @ features_text.text_embeds_proj[:,0,:].t()
print(similarity)
