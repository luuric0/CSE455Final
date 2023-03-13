# CSE 455 Kaggle Final
## Richard Luu and Hao Le
### Video Link: 

## Overview
For our final project, we decided to participate in the biannual Bird Classification Competition on Kaggle. The goal of this competition is to classify the name of each bird accordingly to the image with the highest classification accuracy possible. For our work, we used Juypter and pytorch as the framework. With its extensive library, we were able to perform the training and produce our prediction. The neural network we used for the classification was the ResNet neural network architecture specifically, we used ResNet18 and Resnet152. 

## Starting our training 
For our project, we used the Jupyter notebook to run the code and a rented GPU from vast.ai to train our model. We used the python library PyTorch as the framework to develop our neural network for the competition.

```
import numpy as np
import matplotlib.pyplot as plt
import os

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

The GPU used in the training was a GTX 4090.

For our dataset, we used the provided data set provided by Kaggle. The dataset consists of 10000 images of various birds and their respective name. In our workspace, we used a series of command lines to unpack our uploaded kaggle.json file to generate the dataset.

```
Import Kaggle Dataset

! pip install -q kaggle
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! kaggle competitions download -c birds23wi
! mkdir 'checkpoints'

# Create checkpoints folder to save models
checkpoints = '/workspace/checkpoints'
```

## Neural Network and Training

The notebook we created to train was adapted from the class tutorial code. With the code used, we fine-tuned and calibrated it to what felt appropriate to get the best result. 

### ResNet18

The model starts by loading the data for training, which is done with the existing code from the class tutorial. We wanted to have a baseline prediction, which we determine that keeping the values from the tutorial would be the best approach. 

```
# Method to load datasets

def get_bird_data(augmentation=0):
    transform_train = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomCrop(128, padding=8, padding_mode='edge'), # Take 128x128 crops from padded images
        transforms.RandomHorizontalFlip(),    # 50% of time flip image along y-axis
        transforms.ToTensor(),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.ImageFolder(root='/workspace/birds23wi/birds/train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.ImageFolder(root='/workspace/birds23wi/birds/test', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    classes = open("/workspace/birds23wi/birds/names.txt").read().strip().split("\n")
    class_to_idx = trainset.class_to_idx
    idx_to_class = {int(v): int(k) for k, v in class_to_idx.items()}
    idx_to_name = {k: classes[v] for k,v in idx_to_class.items()}
    return {'train': trainloader, 'test': testloader, 'to_class': idx_to_class, 'to_name':idx_to_name}

data = get_bird_data()
```

To begin the training we used the pre-trained model of ResNet18, which is then loaded into pytorch to prepare the model for training. From our data, the size set was 128x128 with the training batch size being 128 images.

For the ResNet18 model, we followed the tutorial and went through two sets of training. Our initial set was achieved by loading a pre-trained version of ResNet18 from the PyTorch library. This was then followed by using our current model in a second round of training. After two sets of training, we used the model as our submitted prediction. 

The tests were a total of 10 epochs with each test run for 5 epochs with a learning rate of 0.01. 
