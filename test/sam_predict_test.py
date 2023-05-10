import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def mask2bbox(mask):
    """
    将 mask 转为 bbox

    :param mask: 掩码矩阵
    :return:  bbox x、y、w、h 信息
    """
    mask = np.array(mask, np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    areas = []
    for contour in contours:
        area = cv2.contourArea(contour)
        areas.append(area)
    idx = areas.index(np.max(areas))
    x, y, w, h = cv2.boundingRect(contours[idx])
    bbox = [x, y, w, h]
    return bbox


def mask2poly(mask):
    """
    将 mask 转为坐标

    :param mask: paddleseg 预测图片的输出的 mask
    :return: 二维坐标数组
    """
    mask = np.ascontiguousarray(mask)
    contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    polys = []
    polys_set = []
    for contour in contours:
        polys_set.append(contour.flatten().tolist())
    for idx in range(0, len(polys_set[0]), 2):
        polys.append([polys_set[0][idx], polys_set[0][idx + 1]])
    return polys


image = cv2.imread('./images/20230508174001.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('on')
plt.show()

import sys

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "../models/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
predictor.set_image(image)

# 南瓜 233,233
# 烧饼 257,709
input_point = np.array([[257, 709]])
input_label = np.array([1])
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.show()

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)
print(masks.shape)

for i, (mask, score) in enumerate(zip(masks, scores)):
    print(mask2bbox(mask))
    print(mask2poly(mask))
    print('------------------')
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()

"""
masks, scores, logits = predictor.predict(multimask_output=True)
print(masks.shape)

for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()
"""
