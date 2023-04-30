# Code is adapted from https://medium.com/@stepanulyanin/grad-cam-for-resnet152-network-784a1d65f3 (Accessed 30/04/2023)

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch.utils.model_zoo as model_zoo
from TripletResnetSoftmax import Triplet_ResNet_Softmax, Bottleneck
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

      
# Get image
img = Image.open('data/cows/cow.jpg')
size = (224, 224)
old_size = img.size
ratio = float(size[0])/max(old_size)
new_size = tuple([int(x*ratio) for x in old_size])
img = img.resize(new_size, Image.ANTIALIAS)
new_img = Image.new("RGB", (size[0], size[1]))
new_img.paste(img, ((size[0]-new_size[0])//2, (size[1]-new_size[1])//2))
new_img = np.array(new_img, dtype=np.uint8)
img = new_img.transpose(2, 0, 1)
img = torch.from_numpy(img).float().unsqueeze(0)

# Initialise model
model = Triplet_ResNet_Softmax(Bottleneck, [3, 4, 6, 3], num_classes=182)
weights_imagenet = model_zoo.load_url(model_urls['resnet50'])
weights_imagenet["fc_softmax.weight"] = weights_imagenet["fc.weight"]
weights_imagenet["fc_softmax.bias"] = weights_imagenet["fc.bias"]
weights_imagenet["fc_embedding.weight"] = weights_imagenet["fc.weight"]
weights_imagenet["fc_embedding.bias"] = weights_imagenet["fc.bias"]
model.fc = nn.Linear(2048, 1000)
model.fc_embedding = nn.Linear(1000, 128)
model.fc_softmax = nn.Linear(1000, 182)

# Load model weights and put into eval mode
state = torch.load('PATH/best_model_state.pkl', map_location=torch.device('cpu'))
model.load_state_dict(state['model_state'], strict=False)
model = model.eval()

# Forward pass through network
pred = model(img, img, img)
print(pred.argmax(dim=1))

# Get gradients using prediction
pred[:,0].backward()
gradients = model.get_gradient()
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

# Get activations of last conv layer, weight with gradients
activations = model.get_activations(img).detach()
for i in range(512):
    activations[:, i, :, :] *= pooled_gradients[i]

# Make heatmap
heatmap = torch.mean(activations, dim=1).squeeze()
heatmap = np.maximum(heatmap, 0)
heatmap /= torch.max(heatmap)
plt.matshow(heatmap.squeeze())
plt.savefig('./heatmap.jpg')

# Superimpose heatmap on depth image
img = cv2.imread('./data/cows/cow.jpg')
heatmap1 = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
heatmap1 = np.uint8(255 * heatmap1)
heatmap1 = cv2.applyColorMap(heatmap1, cv2.COLORMAP_JET)
superimposed_img = heatmap1 * 0.4 + img
cv2.imwrite('./depthmap.jpg', superimposed_img)

# Superimpose heatmap on rgb image
img = cv2.imread('./data/rgb.jpg')
heatmap2 = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
heatmap2 = np.uint8(255 * heatmap2)
heatmap2 = cv2.applyColorMap(heatmap2, cv2.COLORMAP_JET)
superimposed_img = heatmap2 * 0.4 + img
cv2.imwrite('./RGBmap.jpg', superimposed_img)