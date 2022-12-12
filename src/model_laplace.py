#!/usr/bin/env python
# coding: utf-8

# We are applying ReLU activation and Max-Pooling after every convolutional layer.
# Before the fully connected layer, we are applying dropout once.
# For the final fully connected layer, we are not applying any activation, as we directly need the regressed coordinates for the keypoints.

# In[1]:


import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import os
import torch
from copy import deepcopy

class FaceKeypointModel(nn.Module):
    def __init__(self, freeze_resnet = False):
        super(FaceKeypointModel, self).__init__()
        
        # Convert 1 filter 3 filter because resnet accepts 3 filter only
        self.conv1 = nn.Conv2d( in_channels=1, out_channels=3, kernel_size=(3, 3), stride=1, padding=1, padding_mode='zeros' )
        
        # Resnet Architecture
        self.resnet18 = models.resnet18(pretrained=True)
        if freeze_resnet:
            for param in self.resnet18.parameters():
                param.requires_grad = False
        # replacing last layer of resnet
        # by default requires_grad in a layer is True
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, 384) 

        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(384, 30) 
        self.variance = nn.Linear(384,1) 
        
    def forward(self, x):
        y0 = self.conv1(x)
        y1 = self.resnet18(y0)
        y_relu = self.relu(y1)
        out= self.linear1(y_relu)
        scale = F.softplus(self.variance(y_relu))
        return out,scale
import torch
class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(
        self, optimizer, patience=5, min_lr=1e-6, factor=0.5
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)   
        
        
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
                
                
                
                
def LaplaceNLLLoss(input, target, scale, eps=1e-06, reduction='mean'):
  loss = torch.log(2*scale) + torch.abs(input - target)/scale

  # Inputs and targets much have same shape
  input = input.view(input.size(0), -1)
  target = target.view(target.size(0), -1)
  if input.size() != target.size():
      raise ValueError("input and target must have same size")

  # Second dim of scale must match that of input or be equal to 1
  scale = scale.view(input.size(0), -1)
  if scale.size(1) != input.size(1) and scale.size(1) != 1:
      raise ValueError("scale is of incorrect size")

  # Check validity of reduction mode
  if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
      raise ValueError(reduction + " is not valid")

  # Entries of var must be non-negative
  if torch.any(scale < 0):
      raise ValueError("scale has negative entry/entries")

  # Clamp for stability
  scale = scale.clone()
  with torch.no_grad():
      scale.clamp_(min=eps)

  # Calculate loss (without constant)
  loss = (torch.log(2*scale) + torch.abs(input - target) / scale).view(input.size(0), -1).sum(dim=1)


  # Apply reduction
  if reduction == 'mean':
      return loss.mean()
  elif reduction == 'sum':
      return loss.sum()
  else:
      return loss