#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[2]:


ROOT_PATH = '/home/nandhini/facial recog/input/facial-keypoints-detection'
OUTPUT_PATH = '/home/nandhini/facial recog/outputs'


# In[4]:


BATCH_SIZE = 256
LR = 0.0001
EPOCHS = 300
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#train test split
TEST_SPLIT = 0.2


# In[5]:


#how dataset keypoint plot
SHOW_DATASET_PLOT = True

