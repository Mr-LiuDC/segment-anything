import cv2
import numpy as np
import torch

from segment_anything import SamPredictor, sam_model_registry

checkpoint = "../models/sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=checkpoint)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sam.to(device=device)
predictor = SamPredictor(sam)

image = cv2.imread('./images/20230508174001.png')
predictor.set_image(image)
image_embedding = predictor.get_image_embedding().cpu().numpy()
np.save("./images/20230508174001_embedding.npy", image_embedding)
