from torch.hub import download_url_to_file
import torch
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
import os

from comfy.model_management import get_torch_device
DEVICE = get_torch_device()

# Convert PIL to Tensor
# 图片转张量
def pil2tensor(image, device=DEVICE):
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        raise Exception("Input image should be either PIL Image!")

    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))  # chw
        print(f"Prepare the imput images")
    elif img.ndim == 2:
        img = img[np.newaxis, ...]
        print(f"Prepare the imput masks")

    assert img.ndim == 3

    try:
        img = img.astype(np.float32) / 255
    except:
        img = img.astype(np.float16) / 255
    
    out_image = torch.from_numpy(img).unsqueeze(0).to(device)
    return out_image

# Tensor to PIL
# 张量转图片
def tensor2pil(image):
    i = 255. * image.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    return img

# pil to comfy
# 图片转comfy格式 (i, 3, w, h) -> (i, h, w, 3)
def pil2comfy(img):
    img = ImageOps.exif_transpose(img)
    image = img.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    return image

