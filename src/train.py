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

from modelrmse import FaceKeypointModel, EarlyStopping, LRScheduler
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
#criterion = nn.MSELoss()
def RMSE(y_train_pred,Y):
    
    mse = torch.mean(torch.square(y_train_pred - Y))
    rmse = torch.sqrt(mse)
    
    return rmse

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
    train_running_loss = 0.0
    counter = 0
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    for i, data in tqdm(enumerate(dataloader), total=num_batches):
        counter += 1
        image, keypoints = data['image'].to(config.DEVICE), data['keypoints'].to(config.DEVICE)
        # flatten the keypoints
        keypoints = keypoints.view(keypoints.size(0), -1)
        optimizer.zero_grad()
        outputs = model(image)
        loss = RMSE(outputs, keypoints)
        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    train_loss = train_running_loss/counter
    return train_loss


# In[6]:


def validate(model, dataloader, data, epoch):
    print('Validating')
    model.eval()
    valid_running_loss = 0.0
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
            outputs = model(image)
            loss = RMSE(outputs, keypoints)
            valid_running_loss += loss.item()
            val_difference.extend(abs(keypoints - outputs).squeeze(1).tolist())
            # plot the predicted validation keypoints after every...
            # ... 25 epochs and from the first batch
            if (epoch+1) % 25 == 0 and i == 0:
                utils.valid_keypoints_plot(image, outputs, keypoints, epoch)
    diff = np.average(val_difference)    
    valid_loss = valid_running_loss/counter
    return valid_loss,diff


# In[7]:

early_stopping = EarlyStopping()
train_loss = []
val_loss = []
prediction = []
for epoch in range(config.EPOCHS):
    print(f"Epoch {epoch+1} of {config.EPOCHS}")
    train_epoch_loss = fit(model, train_loader, train_data)
    val_func = validate(model, valid_loader, valid_data, epoch)
    val_epoch_loss= val_func[0]
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    pred = val_func[1]
    prediction.append(pred)
    if args['lr_scheduler']:
        lr_scheduler(val_epoch_loss)
    if args['early_stopping']:
        early_stopping(val_epoch_loss)
        if early_stopping.early_stop:
            break
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f'Val Loss: {val_epoch_loss:.4f}')
    print(f'pred: {pred:.4f}')
new = np.vstack((train_loss,val_loss,prediction)).T
np.savetxt("RMSE.csv", new, delimiter=",")

# In[8]:


# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"{config.OUTPUT_PATH}/loss.png")
plt.show()
filename = "/home/nandhini/NandhiniMathivananRnD/outputs/model"
torch.save(model.state_dict(), filename)
#torch.save({
#            'epoch': config.EPOCHS,
#            'model_state_dict': model.state_dict(),
#            'optimizer_state_dict': optimizer.state_dict(),
#            'loss': RMSE,
#            }, f"{config.OUTPUT_PATH}/model")
print('DONE TRAINING')


# In[ ]:




