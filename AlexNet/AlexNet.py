import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time


# Argument configuration
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", type = int, choices=[0, 1],
		 help = "enables gpu mode for inference 0 (default) for CPU mode and 1 for GPU mode",
		 default = 0)
parser.add_argument("-f", "--file", type = str,
		 help = "image file for inference testing by default cat.jpg sample",
		 default = "cat.jpg")
args = parser.parse_args()


# Using pretained model for Forward pass evaluation in platform
alexNet = models.alexnet(pretrained = True)
alexNet.eval()

# Check for CUDA availability
if(torch.cuda.is_available() & args.gpu):
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


# Normalize and Resize input 
loader = transforms.Compose([transforms.Resize(size=(224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
input = Image.open(args.file,'r')
plt.imshow(np.asarray(input))
input = loader(input)
# Change to BatchxChannelxHeightxWidth (1xCxHxW) for inference
input = input.unsqueeze(0)

# Test inference time on a single input image on GPU or CPU 
if(torch.cuda.is_available() & args.gpu):
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
