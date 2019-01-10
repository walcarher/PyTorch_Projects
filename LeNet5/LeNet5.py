import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	# Definition of the Conv model (LeNet5)
	def __init__(self):
		super(Net, self).__init__()
		# 1 input image, 6 output channels, kernel size of 5x5
		self.conv1 = nn.Conv2d(1, 6, 5)
		# 6 input feature (must match previous layer output channels), 16 output channels, 			kernel size of 5x5, if no stride provided 1 by 1 is assumed
		self.conv2 = nn.Conv2d(6, 16, 5)
		# Linear function y = mx + b, Unroll previous 5x5 kernel size and previous layer 			output channel. Fully connected layer definitions
		# In this example the fully connected part model is defined into the GPU
		# while the rest is run on the CPU (Example of heterogeneous execution)
		# Only activation betweeb Conv2 and FC1 are passed into the GPU memory 
		self.fc1 = nn.Linear(5*5*16,120).cuda()
		self.fc2 = nn.Linear(120,84).cuda()
		self.fc3 = nn.Linear(84,10).cuda()

	# Activations and pooling are executed here (forward pass before backward pass)	
	def forward(self, x, dev):
		# Max pooling from functional library on ReLu also from functional on the first 		convolutional layer, size of max pooling window 2x2
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		# If windows is square a single parameter is enough
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = x.view(-1, self.num_flat_features(x)) ### ??? Look up for this line!!! Must be 			something for compatibility between conv layers and fc layers (Unroll=flat?)
		# This is where the Heterogeneous systems comes into de game
		# In this example fully connected layers are running on the GPU
		x = x.to(device = dev)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def num_flat_features(self, x):
		# All dimensions from network except batches
 		size = x.size()[1:]
		# Initialize flattened featuresa 
		num_features = 1
		# Multiplicator
		for s in size:
			num_features *= s
		return num_features

# Check GPU availability, computing capabilities and device 
print('Is CUDA available? : ',torch.cuda.is_available())
if torch.cuda.is_available():
	device = torch.cuda.current_device()
	print('With device : ',torch.cuda.get_device_name(device))
	print('With device capability: ',torch.cuda.get_device_capability(device)) 

# Print CNN layers and input/output features
net = Net()
print(net)

# Print parameters 
params = list(net.parameters())
print(len(params))
print(params[0].size())
#print(params[0])

input = torch.rand(1,1,32,32)
out = net(input, device)
print(out)


	
		
		
			
