import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

import sys
sys.path.append("..")
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "./weights/mobile_sam.pt"
model_type = "vit_t"

device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
sam.eval()

mask_generator = SamAutomaticMaskGenerator(sam)


city_classes = [
    'road', 'sidewalk', 'parking', 'rail track',
    'person', 'rider',
    'car', 'truck', 'bus', 'on rails', 'motorcycle', 'bicycle', 'caravan', 'trailer',
    'building', 'wall', 'fence', 'guard rail', 'bridge', 'tunnel',
    'pole', 'pole group', 'traffic sign', 'traffic light',
    'vegetation', 'terrain',
    'sky',
    'ground', 'dynamic', 'static'
]

import clip
import os
from tqdm.auto import tqdm
from PIL import Image

def generate_text_embeddings(classnames, templates, model):
    with torch.no_grad():
        class_embeddings_list = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).to(device) #tokenize
            class_embedding = model.encode_text(texts) #embed with text encoder
            class_embeddings_list.append(class_embedding)
        class_embeddings = torch.stack(class_embeddings_list, dim=1).to(device)
    return class_embeddings

# with torch.no_grad():
#     predict()

## clip
def predict():
    global image
    clip_model, preprocess = clip.load("ViT-B/16", device=device)
    text_features = generate_text_embeddings(city_classes, ['a clean origami {}.'], clip_model)#['a rendering of a weird {}.'], model)
    folder_path = "./data/leftImg8bit/val/frankfurt"
    png_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]
    for png_file in png_files:
        # Construct the full path to the PNG file
        file_path = os.path.join(folder_path, png_file)

        image = np.array(Image.open(file_path).resize((1024, 1024)))
        if image is None:
            print(f"Could not load '{file_path}' as an image, skipping...")
            continue

        masks = mask_generator.generate(image)
        plt.figure(figsize=(20,20))
        plt.imshow(image)
        # show_anns(masks)
        plt.axis('off')
        plt.show() 
            
        clip_model.eval()
        for i in range(len(masks)):
            mask = masks[i]["segmentation"]
            image_new = image.copy()
            ind = np.where(mask > 0)
            image_new[mask == 0] = 0
            y1, x1, y2, x2 = min(ind[0]), min(ind[1]), max(ind[0]), max(ind[1])
            image_new = Image.fromarray(image_new[y1:y2+1, x1:x2+1])
            # plt.imshow(image_new)
            # plt.show()
            image_new = preprocess(image_new)
            # plt.imshow(image_new.permute(1, 2, 0))
            image_features = clip_model.encode_image(image_new.unsqueeze(0).to(device))
            # Pick the top 5 most similar labels for the image
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.squeeze(0)
            similarity = (100.0 * image_features.float() @ text_features.float().T).softmax(dim=-1)
            values, indices = similarity[0].topk(5)

            print("\nTop predictions for {}-th mask:\n".format(i), end='')
            for value, index in zip(values, indices):
                print(f"{city_classes[index]:>16s}: {100 * value.item():.2f}%")
        print("Done!")
# Release memory by deleting variables
    del image, masks, clip_model
    torch.cuda.empty_cache()

with torch.no_grad():
    predict()