#!/usr/bin/python
from detr import DETRsimple
from utils import *

from PIL import Image
import time
import cv2

import torch
import torchvision.transforms as T
torch.set_grad_enabled(False)

# Image path to detect
url = '/home/grvc/Downloads/photo5974074985481876075.jpg'
use_matplotlib = True


# Configure GPU support
if torch.cuda.is_available():  
  dev = "cuda:0" 
  print("Using CUDA with " + torch.cuda.get_device_name(0))
else:  
  dev = "cpu"  

# Load DETR model pre-trained
detr = DETRsimple(num_classes=91)
state_dict = torch.hub.load_state_dict_from_url(
    url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth', check_hash=True)
detr.load_state_dict(state_dict)
detr.eval()
# Move DETR to cuda device
detr = detr.to(dev)

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if not(use_matplotlib):
    cv2.namedWindow("preview")
cap = cv2.VideoCapture(0)

if cap.isOpened(): # try to get the first frame
    rval, frame = cap.read()
else:
    rval = False

if use_matplotlib:
    plt.figure(figsize=(16,10))
    plt.imshow(frame)
    plt.ion()

while rval:
    im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    start = time.time()
    scores, boxes = detect(im, detr, transform, dev)
    end = time.time()
    print("FPS: ", 1.0 / (time.time() - start)) # FPS = 1 / time to process loop

    if use_matplotlib:
        plot_results_dyn(plt, im, scores, boxes)
    else:
        im2 = plot_bboxes_cv2(im, scores, boxes)
        cv2.imshow("preview", im2)

    rval, frame = cap.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cap.release()

if use_matplotlib:
    plt.ioff()
    plt.show()
else:
    cv2.destroyWindow("preview")