#!/usr/bin/python

from detr import DETRsimple
from utils import *
from PIL import Image
import time
import math
import torch
import torchvision.transforms as T
torch.set_grad_enabled(False)

# Image path to segment
url = '/home/grvc/Downloads/photo5974074985481876075.jpg'
plot_beauty = True

# Configure GPU support
if torch.cuda.is_available():  
  dev = "cuda:0" 
  print("Using CUDA with " + torch.cuda.get_device_name(0))
else:  
  dev = "cpu"  

# Detectron2 uses a different numbering scheme, we build a conversion table
coco2d2 = {}
count = 0
for i, c in enumerate(CLASSES):
  if c != "N/A":
    coco2d2[i] = count
    count+=1

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model, postprocessor = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True, return_postprocessor=True, num_classes=250)
model.eval()

# Move DETR to cuda device
model = model.to(dev)

# Load image
im = Image.open(url)
start = time.time()

# mean-std normalize the input image (batch-size: 1)
img = transform(im).unsqueeze(0).cuda()
out = model(img)

end = time.time()
print('Segmentation time: '+ str(end - start) + ' sec')

# compute the scores, excluding the "no-object" class (the last one)
scores = out["pred_logits"].softmax(-1)[..., :-1].max(-1)[0]
# threshold the confidence
keep = scores > 0.85

# DEBUG -- Print mask that DETR segment individually
#plot_masks_subplot(out,keep)

# the post-processor expects as input the target size of the predictions (which we set here to the image size)
result = postprocessor(out, torch.as_tensor(img.shape[-2:]).unsqueeze(0).cuda())[0]

# Plot all masks together
if plot_beauty:
  segm_img = plot_segmentation_masks_beauty(result, im)
  print('\nClose figure using "Enter" pls')
  cv2.imshow("panoptic",segm_img)
  cv2.waitKey(0)
  cv2.destroyWindow("panoptic")
else:
  panoptic_img = plot_segmentation_masks(result)
  plt.figure(figsize=(15,15))
  plt.imshow(panoptic_img)
  plt.axis('off')
  plt.show()
