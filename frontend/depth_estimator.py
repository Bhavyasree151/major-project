# depth_estimator.py
import torch
import cv2
import numpy as np

# Load MiDaS small model and transforms
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to('cuda' if torch.cuda.is_available() else 'cpu').eval()

transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

def estimate_depth(image):
    input_tensor = transform(image).to('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bilinear",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    return depth_map
