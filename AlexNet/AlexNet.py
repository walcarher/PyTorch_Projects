import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time


# --gpu-enable 1 (True) 0 (False) Runs the whole model and input on main GPU device 
if(sys.argv[1] == '--gpu-enable'):
	if(sys.argv[2] == '1'):
		cuda_mode = True
	else:
		cuda_mode = False
else:
	print('Error : No argument passed for CPU/GPU running mode')
	sys.exit()

# Using pretained model for Forward pass evaluation in platform
alexNet = models.alexnet(pretrained = True)
alexNet.eval()

# Check for CUDA availability
if(torch.cuda.is_available() & cuda_mode):
	if(not torch.cuda.is_available()): 
		print('Error : No GPU available on the system')
		sys.exit()
	device = torch.device(torch.cuda.current_device())
	torch.cuda.init()
	alexNet.cuda()
	print('Running inference on GPU mode')
else:
	device = torch.device("cpu")
	print('Running inference on CPU mode')


# Normalize input WARNING: Check if input image is really normalized
loader = transforms.Compose([transforms.Resize(size=(224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
input = Image.open('cat.jpg','r')
plt.imshow(np.asarray(input))
input = loader(input)
# Change to BatchxChannelxHeightxWidth (1xCxHxW) for inference
input = input.unsqueeze(0)

# Test inference time on a single input image on GPU or CPU 
if(torch.cuda.is_available() & cuda_mode):
	# Wait until device is finished on every Kernel
	torch.cuda.synchronize()
	start = time.time()
	out = alexNet.forward(input.cuda())
	end = time.time()
	torch.cuda.synchronize()
	print(torch.argmax(out[0,:]))
	print('Inference Time on GPU : %.4gs'%(end-start))
else:
	start = time.time()
	out = alexNet.forward(input)
	end = time.time()
	print(torch.argmax(out[0,:]))
	print('Inference Time on CPU : %.4gs'%(end-start))

plt.show()
