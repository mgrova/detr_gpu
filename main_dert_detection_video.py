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

cv2.namedWindow("preview")
cap = cv2.VideoCapture(0)

if cap.isOpened(): # try to get the first frame
    rval, frame = cap.read()
else:
    rval = False

while rval:
    im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    start = time.time()
    scores, boxes = detect(im, detr, transform, dev)
    end = time.time()
    print('Detection time: '+ str(end - start) + ' sec')
    im2 = plot_bboxes(im, scores, boxes)

    cv2.imshow("preview", im2)

    rval, frame = cap.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
cap.release()
cv2.destroyWindow("preview")

# # Load image
# im = Image.open(url)
# 
# # Detect in image and measure time
# start = time.time()
# scores, boxes = detect(im, detr, transform, dev)
# end = time.time()
# print('Detection time: '+ str(end - start) + ' sec')

#Plot results of detection    