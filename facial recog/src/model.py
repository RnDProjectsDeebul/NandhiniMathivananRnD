#!/usr/bin/env python
# coding: utf-8

# We are applying ReLU activation and Max-Pooling after every convolutional layer.
# Before the fully connected layer, we are applying dropout once.
# For the final fully connected layer, we are not applying any activation, as we directly need the regressed coordinates for the keypoints.

# In[1]:


import torch.nn as nn
import torch.nn.functional as F
class FaceKeypointModel(nn.Module):
    def __init__(self):
        super(FaceKeypointModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(128, 30) 
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(p=0.2)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = self.dropout(x)
        out = self.fc1(x) 
        return out

