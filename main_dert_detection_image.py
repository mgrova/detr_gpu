#!/usr/bin/python

from detr import DETRsimple
from utils import *
from PIL import Image
import time
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

# Load image
im = Image.open(url)

# Detect in image and measure time
start = time.time()
scores, boxes = detect(im, detr, transform, dev)
end = time.time()
print('Detection time: '+ str(end - start) + ' sec')

#Plot results of detection    
plot_results(im, scores, boxes)