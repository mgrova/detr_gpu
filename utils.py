#!/usr/bin/python

import cv2
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import itertools
import seaborn as sns
import io
from PIL import Image

import panopticapi
from panopticapi.utils import id2rgb, rgb2id
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from copy import deepcopy

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# ----------------------------------------------------------------------------------
# --------------------------- DETECTION UTILS --------------------------------------
# ----------------------------------------------------------------------------------

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

# ----------------------------------------------------------------------------------
def rescale_bboxes(out_bbox, size, dev):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(dev)
    return b

# ----------------------------------------------------------------------------------
def detect(im, model, transform, dev):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size, dev)
    return probas[keep], bboxes_scaled

# ----------------------------------------------------------------------------------
def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

# ----------------------------------------------------------------------------------
def plot_results_dyn(plt, pil_img, prob, boxes):
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.imshow(pil_img)
    plt.pause(0.0001)
    plt.clf()

# ----------------------------------------------------------------------------------
def plot_bboxes_cv2(pil_img, prob, boxes):

    numpy_image=np.array(pil_img)  

    # convert to a openCV2 image, notice the COLOR_RGB2BGR which means that 
    # the color is converted from RGB to BGR format
    opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR) 
    
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        init_pt = (int(xmin),int(xmax))
        end_pt = (int(xmax - xmin), int(ymax - ymin))
        c1 = [int(x*255) for x in c]
        
        cv2.rectangle(opencv_image, init_pt, end_pt, c1, 3)
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        cv2.putText(opencv_image, text,init_pt,cv2.FONT_HERSHEY_COMPLEX,0.5,c1,1)
    return opencv_image


# ----------------------------------------------------------------------------------
# --------------------------- SEGMENTATION UTILS -----------------------------------
# ----------------------------------------------------------------------------------
def plot_masks_subplot(image, keep):
    # Plot all the remaining masks
    ncols = 5
    fig, axs = plt.subplots(ncols=ncols, nrows=math.ceil(keep.sum().item() / ncols), figsize=(18, 10))
    for line in axs:
        for a in line:
            a.axis('off')
    for i, mask in enumerate(image["pred_masks"][keep]):
        ax = axs[i // ncols, i % ncols]
        ax.imshow(mask.cpu(), cmap="cividis")
        ax.axis('off')

    fig.tight_layout()
    plt.show()

# ----------------------------------------------------------------------------------
def plot_segmentation_masks(image):
    palette = itertools.cycle(sns.color_palette())

    # The segmentation is stored in a special-format png
    panoptic_seg = Image.open(io.BytesIO(image['png_string']))
    panoptic_seg = np.array(panoptic_seg, dtype=np.uint8).copy()
    # We retrieve the ids corresponding to each mask
    panoptic_seg_id = rgb2id(panoptic_seg)

    # Finally we color each mask individually
    panoptic_seg[:, :, :] = 0
    for id in range(panoptic_seg_id.max() + 1):
        panoptic_seg[panoptic_seg_id == id] = np.asarray(next(palette)) * 255
    plt.figure(figsize=(15,15))
    plt.imshow(panoptic_seg)
    plt.axis('off')
    plt.show()

# ----------------------------------------------------------------------------------
def plot_segmentation_masks_beauty(result, image):
    # We extract the segments info and the panoptic result from DETR's prediction
    segments_info = deepcopy(result["segments_info"])
    # Panoptic predictions are stored in a special format png
    panoptic_seg = Image.open(io.BytesIO(result['png_string']))
    final_w, final_h = panoptic_seg.size
    # We convert the png into an segment id map
    panoptic_seg = np.array(panoptic_seg, dtype=np.uint8)
    panoptic_seg = torch.from_numpy(rgb2id(panoptic_seg))
        
    # Detectron2 uses a different numbering of coco classes, here we convert the class ids accordingly
    meta = MetadataCatalog.get("coco_2017_val_panoptic_separated")
    for i in range(len(segments_info)):
        c = segments_info[i]["category_id"]
        segments_info[i]["category_id"] = meta.thing_dataset_id_to_contiguous_id[c] if segments_info[i]["isthing"] else meta.stuff_dataset_id_to_contiguous_id[c]

    # Finally we visualize the prediction
    v = Visualizer(np.array(image.copy().resize((final_w, final_h)))[:, :, ::-1], meta, scale=1.0)
    v._default_font_size = 20
    v = v.draw_panoptic_seg_predictions(panoptic_seg, segments_info, area_threshold=0)
    print('\nClose figure using "Enter" pls')
    cv2.imshow("panoptic",v.get_image())
    cv2.waitKey(0)
    cv2.destroyWindow("panoptic")

# ----------------------------------------------------------------------------------