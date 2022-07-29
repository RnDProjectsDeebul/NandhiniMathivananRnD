#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib
import config
import utils

from model_laplace import FaceKeypointModel, EarlyStopping, LRScheduler,LaplaceNLLLoss
from dataset import train_data, train_loader, valid_data, valid_loader
from tqdm import tqdm
matplotlib.style.use('ggplot')
import argparse
import numpy as np
# In[4]:
parser = argparse.ArgumentParser()
parser.add_argument('--lr-scheduler', dest='lr_scheduler', action='store_true')
parser.add_argument('--early-stopping', dest='early_stopping', action='store_true')
args = vars(parser.parse_args())

# model 
model = FaceKeypointModel().to(config.DEVICE)
# optimizer
optimizer = optim.Adam(model.parameters(), lr=config.LR)
# we need a loss function which is good for regression like MSELoss



if args['lr_scheduler']:
    print('INFO: Initializing learning rate scheduler')
    lr_scheduler = LRScheduler(optimizer)
    # change the accuracy, loss plot names and model name
    loss_plot_name = 'lrs_loss'
    acc_plot_name = 'lrs_accuracy'
    model_name = 'lrs_model'
if args['early_stopping']:
    print('INFO: Initializing early stopping')
    early_stopping = EarlyStopping()
    # change the accuracy, loss plot names and model name
    loss_plot_name = 'es_loss'
    acc_plot_name = 'es_accuracy'
    model_name = 'es_model'

# In[5]:


def fit(model, dataloader, data):
    print('Training')
    model.train()
    train_running_loss = []
    
    
    counter = 0
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    for i, data in tqdm(enumerate(dataloader), total=num_batches):
        counter += 1
        image, keypoints = data['image'].to(config.DEVICE), data['keypoints'].to(config.DEVICE)
        # flatten the keypoints
        keypoints = keypoints.view(keypoints.size(0), -1)
        optimizer.zero_grad()
        outputs,scale = model(image)
        loss = LaplaceNLLLoss(outputs, keypoints,scale)
        #train_running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()
        train_running_loss.append(loss.tolist())
        
        
    train_loss = np.average(train_running_loss)
  
    return train_loss


# In[6]:


def validate(model, dataloader, data, epoch):
    print('Validating')
    model.eval()
    valid_running_loss = []
    Scale = []
    val_difference = []
    
    counter = 0
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=num_batches):
            counter += 1
            image, keypoints = data['image'].to(config.DEVICE), data['keypoints'].to(config.DEVICE)
            # flatten the keypoints
            keypoints = keypoints.view(keypoints.size(0), -1)
            outputs,scale = model(image)
            loss = LaplaceNLLLoss(outputs, keypoints,scale)
            #valid_running_loss += loss.item()
            valid_running_loss.append(loss.tolist())
            Scale.extend(scale.tolist())
            val_difference.extend(abs(keypoints - outputs).squeeze(1).tolist())
            # plot the predicted validation keypoints after every...
            # ... 25 epochs and from the first batch
            if (epoch+1) % 25 == 0 and i == 0:
                utils.valid_keypoints_plot(image, outputs, keypoints, epoch)
    valid_loss = np.average(valid_running_loss)
    avg_scale  = np.average(Scale)  
    diff = np.average(val_difference)
    
    
    

    return valid_loss, avg_scale,diff




train_loss = []
val_loss = []
finalscale = []
prediction = []
#entropy=[]

for epoch in range(config.EPOCHS):
    print(f"Epoch {epoch+1} of {config.EPOCHS}")
    train_func = fit(model, train_loader, train_data)
    train_epoch_loss = train_func
    val_epoch_loss = validate(model, valid_loader, valid_data, epoch)
    Val_loss = val_epoch_loss[0]
    lap_scale = val_epoch_loss[1]
    pred = val_epoch_loss[2]
#    ent = 0.5*(np.log(2*np.pi*Variance))+1/2
    train_loss.append(train_epoch_loss)
    val_loss.append(Val_loss)
    finalscale.append(lap_scale)
    prediction.append(pred)
#    entropy.append(ent)
    if args['lr_scheduler']:
        lr_scheduler(Val_loss)
    if args['early_stopping']:
        early_stopping(Val_loss)
        if early_stopping.early_stop:
            break
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f'Val Loss: {Val_loss:.4f}')
    print(f'scale: {lap_scale:.4f}')
    print(f'pred: {pred:.4f}')
new = np.vstack((train_loss,val_loss,prediction)).T
np.savetxt("laplace.csv", new, delimiter=",")

# In[8]:


plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"{config.OUTPUT_PATH}/loss.png")
plt.show()
filename = "/home/nandhini/NandhiniMathivananRnD/outputs/lap_model"

torch.save(model.state_dict(), filename)
#torch.save({
#            'epoch': config.EPOCHS,
#            'model_state_dict': model.state_dict(),
#            'optimizer_state_dict': optimizer.state_dict(),
#            'loss': LaplaceNLLLoss,
#            }, f"{config.OUTPUT_PATH}/lap_model")
print('DONE TRAINING')



# In[ ]:




