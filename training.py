# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 11:46:41 2024

@author: 110511185 張佑禕
"""

# float digit preference
digits = 4

from tqdm import tqdm
import numpy as np
import cv2

# core: torch neural network
import torch
import torch.nn as nn
import torch.utils.data as data



from torchvision.datasets import ImageFolder as ImgF
from torchvision import transforms

# import os to show cuda path and checking model path
import os

# import torchstat to show model properties
from torchstat import stat

# plot
import matplotlib.pyplot as plt

model_path = "./model_params.pth"

print()

print("torch     version   " + str(torch.__version__))
print("required     cuda   " + str(torch.version.cuda))
print()

print("Hardware:")
print("GPU       version   " + str(torch.cuda.get_device_name(0)))
print()

print("cuda")
print("cuda path and version")
print("    " + str(os.environ.get('CUDA_PATH')))
print("cuda    available   " + str(torch.cuda.is_available()))
print()

print("cuDNN")
print("cuDNN     version   " + str(torch.backends.cudnn.version()))
print("cuDNN     enabled   " + str(torch.backends.cudnn.enabled))
print()

# set GPU/CPU device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device chosen: " + str(device))

def setdigit(x):
    return "{:10." + str(x) + "f}"




# random rotation limit
deg = 5

# define transform process
Trans = transforms.Compose([
    transforms.Resize([40, 40]),
    transforms.RandomRotation(deg),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

# define transform process
Trans_test = transforms.Compose([
    transforms.Resize([40, 40]),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

# create train/test dataset
train = ImgF(root='./balanced_train', transform=Trans)
test  = ImgF(root='./balanced_test', transform=Trans_test)
print("dataset ready")

# create loaders
batch = 500
trainloader = data.DataLoader(train, batch_size=batch, shuffle=True,  num_workers=0)
testloader  = data.DataLoader(test,  batch_size=50,    shuffle=False, num_workers=0) 
print("loader ready")

# CNN structure

# input shape
input_shape = (-1,3,40,40)
class CNN(nn.Module):
    # initialization
    def __init__(self):
        super(CNN, self).__init__()

        self.cin    = nn.Conv2d(in_channels= 3, out_channels=4, kernel_size=5, stride=1, padding=0)
        # hidden convolution
        self.c1_1   = nn.Conv2d(in_channels= 4, out_channels= 8,  kernel_size=5, stride=1, padding=0)
        self.c1_2   = nn.Conv2d(in_channels= 8, out_channels=16,  kernel_size=3, stride=1, padding=0)
        
        self.c2_1   = nn.Conv2d(in_channels=16, out_channels=32,  kernel_size=3, stride=1, padding=0)
        self.c2_2   = nn.Conv2d(in_channels=32, out_channels=32,  kernel_size=3, stride=1, padding=0)

        # Max pool, kernel = 2
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2) 
        # Avg pool, kernel = 2
        self.avgpool2 = nn.AvgPool2d(kernel_size = 2)

        # output fc layer
        self.fc = nn.Linear(800, 28, bias = True)

        # activation
        self.relu    = nn.ReLU()
        

    def forward(self, x):
        x = self.cin(x)
        
        x = self.c1_1(x)
        x = self.c1_2(x)

        x = self.maxpool2(x)

        x = self.c2_1(x)
        x = self.c2_2(x)
        x = self.maxpool2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.fc(x)
        return x

  

# fitting
def fitting(model, loss_func, optimizer, input_shape, num_epochs, train_loader, test_loader):
    model = model.to(device)
    # training accu for recorded model
    train_accu_rec = 50
    # testing accu for recorded model
    test_accu_rec = 50
    
    # overfitting rate (training accu - testing accu) tolerence in %
    overfitting_tol = 15
    
    # Training the Model
    training_loss = []
    training_accuracy = []
    testing_loss = []
    testing_accuracy = []
    for epoch in range(num_epochs):
        #pre-testing before training
        #evaluate model & store loss & acc / epoch
        if(epoch == 0):
            correct_train = 0
            total_train = 0
            for i, (images, labels) in enumerate(train_loader):
                # 1.Define variables
                # remember to move them onto the demanded device
                train  = images.view(input_shape).clone().detach().to(device)
                labels = labels.clone().detach().to(device)
                # 2.Forward propagation
                outputs = model(train)
                # 3.Cross entropy loss
                train_loss = loss_func(outputs, labels)
                # 4.Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]
                # 5.Total number of labels
                total_train += len(labels)
                # 6.Total correct predictions
                correct_train += (predicted == labels).float().sum()
            # store testing_acc for each epoch
            # remember, each elem is a cuda tensor, so put them back to cpu before saved into a list
            # a list can't directly moved to cpu like list.cpu()
            train_accu_rec = 100 * correct_train / float(total_train)
        
            correct_test = 0
            total_test = 0
            for i, (images, labels) in enumerate(test_loader):
                # 1.Define variables
                # remember to move them onto the demanded device
                test   = images.view(input_shape).clone().detach().to(device)
                labels = labels.clone().detach().to(device)
                # 2.Forward propagation
                outputs = model(test)
                # 3.Cross entropy loss
                test_loss = loss_func(outputs, labels)
                # 4.Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]
                # 5.Total number of labels
                total_test += len(labels)
                # 6.Total correct predictions
                correct_test += (predicted == labels).float().sum()
            # store testing_acc for each epoch
            # remember, each elem is a cuda tensor, so put them back to cpu before saved into a list
            # a list can't directly moved to cpu like list.cpu()
            test_accu_rec = 100 * correct_test / float(total_test)
            
            print("  Initial")
            print("  Training accu: " + str(setdigit(digits).format(train_accu_rec))   + " %")  
            print("  Testing  accu: " + str(setdigit(digits).format(test_accu_rec))    + " %")
            print()
            
        #training model & store loss & acc / epoch
        correct_train = 0
        total_train = 0
        for i, (images, labels) in enumerate(train_loader):
            # define var.
            # remember to move them onto the demanded device
            train = images.view(input_shape).clone().detach().requires_grad_(True).to(device)
            labels = labels.clone().detach().to(device)
            # initialize gradient
            optimizer.zero_grad()
            # forward propagation
            outputs = model(train)
            # cross entropy loss
            train_loss = loss_func(outputs, labels)
            # calculate gradients
            train_loss.backward()
            # update parameters
            optimizer.step()
            # predict from the maximum value
            predicted = torch.max(outputs.data, 1)[1]
            # renew number of labels
            total_train += len(labels)
            # renew number correct predictions
            correct_train += (predicted == labels).float().sum()
        
        # store training_acc for each epoch
        # remember, each elem is a cuda tensor, so put them back to cpu before saved into a list
        # a list can't directly moved to cpu like list.cpu()
        train_accuracy = 100 * correct_train / float(total_train)


        #evaluate model & store loss & acc / epoch
        correct_test = 0
        total_test = 0
        for i, (images, labels) in enumerate(test_loader):
            # 1.Define variables
            # remember to move them onto the demanded device
            test   = images.view(input_shape).clone().detach().to(device)
            labels = labels.clone().detach().to(device)
            # 2.Forward propagation
            outputs = model(test)
            # 3.Cross entropy loss
            test_loss = loss_func(outputs, labels)
            # 4.Get predictions from the maximum value
            predicted = torch.max(outputs.data, 1)[1]
            # 5.Total number of labels
            total_test += len(labels)
            # 6.Total correct predictions
            correct_test += (predicted == labels).float().sum()
        # store testing_acc for each epoch
        # remember, each elem is a cuda tensor, so put them back to cpu before saved into a list
        # a list can't directly moved to cpu like list.cpu()
        test_accuracy = 100 * correct_test / float(total_test)

        print("  Trained Epoch: " + str(epoch+1) + "/"        + str(num_epochs))
        print("  Training accu: " + str(setdigit(digits).format(train_accuracy))  + " %")
        print("  Training loss: " + str(setdigit(digits).format(train_loss.data)))
        print("  Testing  accu: " + str(setdigit(digits).format(test_accuracy))   + " %")
        print("  Testing  loss: " + str(setdigit(digits).format(test_loss.data)))
        print()
        
  
        overfitting_rate = train_accuracy.cpu() - test_accuracy.cpu()
        overfitting_rec  = train_accu_rec - test_accu_rec
        
        if(overfitting_rate < overfitting_tol and overfitting_rate > -2):
            if(test_accuracy.cpu() > test_accu_rec):
                torch.save(model, model_path)
                train_accu_rec = train_accuracy.cpu()
                test_accu_rec  = test_accuracy.cpu()
                print("update model due to higher testing accu")
                print()
                
            elif(abs(overfitting_rate) < abs(overfitting_rec)
                 and test_accuracy.cpu() > test_accu_rec - 2):
                torch.save(model, model_path)
                train_accu_rec = train_accuracy.cpu()
                test_accu_rec  = test_accuracy.cpu()
                print("update model due to less overfitting")
                print()

        training_accuracy.append(train_accuracy.cpu())
        # store testing_loss for each epoch
        # remember, each elem is a cuda tensor, so put them back to cpu before saved into a list
        # a list can't directly moved to cpu like list.cpu()
        training_loss.append(train_loss.data.cpu())
        
        testing_accuracy.append(test_accuracy.cpu())
        # store testing_loss for each epoch
        # remember, each elem is a cuda tensor, so put them back to cpu before saved into a list
        # a list can't directly moved to cpu like list.cpu()
        testing_loss.append(test_loss.data.cpu())
        
    tr = training_accuracy
    ts = testing_accuracy
    trl = training_loss
    tsl = testing_loss
    ep = np.linspace(start=1, stop = num_epochs, num = num_epochs)
    
    title_param = "weight decay = " + str(WD) + ", learning rate = " + str(LR)
    
    plt.plot(ep, tr, 'r',  label = "training accu")   
    plt.plot(ep, ts, 'y',  label = "testing  accu")
    plt.title("accuracy when " + title_param)
    plt.legend()
    plt.show()
    
    plt.plot(ep, trl, 'g', label = "training loss")   
    plt.plot(ep, tsl, 'b', label = "testing  loss")
    plt.title("loss when " + title_param)
    plt.legend()
    plt.show()
        
    return training_accuracy[len(training_accuracy) - 1], testing_accuracy[len(testing_accuracy) - 1]
    
# construct model
model = CNN()

# print model properties
print(stat(model, (3, 40, 40)))
print()



# no saved models exist
if(not os.path.exists(model_path)):
    torch.save(model, model_path)
    print("new model created because model not found in recent dir.")    
    print()

# loading saved model if user agreed
else:
    if(input("Load previous model? [y/n] ") == 'n'):
        torch.save(model, model_path)
        print("new model created, covers the previous one")    
        print()
        
    else:
        model = torch.load(model_path)
        



# optimizers
# learning rate: too low results in overfitting
LR = 10 ** -3
# L2 regularization weight decay
WD = 10 ** -1.5

# weighted loss: lower weight for class 'none' for reducing cases that mistaking empty areas for signs
# also set more weight for high speed limit signs for safety
# lookup = np.array(  [0,    1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,    2, 20, 21, 22, 23, 24, 25, 26,   27,    3,    4,    5,    6,    7, 8, 9])
w = torch.FloatTensor([1, 1.01,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1.02,  1,  1,  1,  1,  1,  1,  1, 0.20, 1.03, 1.04, 1.05, 1.06, 1.07, 1, 1])
w = w.to(device)
loss_func = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)   
num_epochs = 40

print("start fitting")
fitting(model, loss_func,  opt, input_shape, num_epochs, trainloader, testloader)
print("training completed")