# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 11:46:41 2024

@author: 110511185 張佑禕
"""

# float digit preference
digits = 4

import numpy as np
import cv2

# core: torch neural network
import torch
import torch.nn as nn
import torch.utils.data as data



from torchvision.datasets import ImageFolder as ImgF
from torchvision import transforms

# FLOPs with torchstat
from torchstat import stat

# import os to show cuda path
import os

# import pyplot for image display
import matplotlib.pyplot as plt
import seaborn as sns


import onnx

# class indices to directory indices
lookup = np.array([0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 3, 4, 5, 6, 7, 8, 9])

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
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
print("device chosen: " + str(device))

def setdigit(x):
    return "{:10." + str(x) + "f}"


def adaptive_thrs(img):
    img32 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    gray  = cv2.cvtColor(img32, cv2.COLOR_BGR2GRAY)
    gray  = cv2.resize(gray, [40, 40])

    # using adaptive threshold to find edges
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # using dilation to remove small black dots
    kernel1 = np.ones((2,2), np.uint8)
    kernel2 = np.ones((1,1), np.uint8)
    di = cv2.dilate(th, kernel1, iterations = 1)
    di = cv2.dilate(th, kernel2, iterations = 3)
 
    """
    cv2.imshow('', di)
    cv2.waitKey(1000)
    """
    return di

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
train = ImgF(root='./balanced_train', transform=Trans_test)
test  = ImgF(root='./balanced_test',  transform=Trans_test)
demo  = ImgF(root='./balanced_val',   transform=Trans_test)

# show matching
print("class name vs. class number")
print(train.class_to_idx)

print("dataset ready")


trainloader = data.DataLoader(train, batch_size=1, shuffle=True,  num_workers=0)
testloader  = data.DataLoader(test,  batch_size=1, shuffle=False, num_workers=0) 
demoloader  = data.DataLoader(demo,  batch_size=1, shuffle=False, num_workers=0) 

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
        
        x = self.fc(x)
        return x

  
# Demo
def Demo(model, loss_func, input_shape, train_loader, test_loader, demo_loader):
    # training accu for recorded model
    train_accu_rec = 50
    # testing accu for recorded model
    test_accu_rec = 50
    # testing accu for recorded model, demo set
    demo_accu_rec = 50
    
    cf_train = np.zeros([28, 28])
    cf_test  = np.zeros([28, 28])
    cf_demo  = np.zeros([28, 28])
    


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
        if(predicted == labels):
            correct_train += 1
        cf_train[lookup[labels], lookup[predicted]] += 1
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
        if(predicted == labels):
            correct_test += 1
        cf_test[lookup[labels], lookup[predicted]] += 1
    # store testing_acc for each epoch
    # remember, each elem is a cuda tensor, so put them back to cpu before saved into a list
    # a list can't directly moved to cpu like list.cpu()
    test_accu_rec = 100 * correct_test / float(total_test)

    correct_demo = 0
    total_demo = 0
    for i, (images, labels) in enumerate(demo_loader):
        # 1.Define variables
        # remember to move them onto the demanded device
        demo   = images.view(input_shape).clone().detach().to(device)
        labels = labels.clone().detach().to(device)
        # 2.Forward propagation
        outputs = model(demo)
        # 3.Cross entropy loss
        demo_loss = loss_func(outputs, labels)
        # 4.Get predictions from the maximum value
        predicted = torch.max(outputs.data, 1)[1]
        # 5.Total number of labels
        total_demo += len(labels)
        # 6.Total correct predictions
        if(predicted == labels):
            correct_demo += 1
        cf_demo[lookup[labels], lookup[predicted]] += 1
    # store testing_acc for each epoch
    # remember, each elem is a cuda tensor, so put them back to cpu before saved into a list
    # a list can't directly moved to cpu like list.cpu()
    demo_accu_rec = 100 * correct_demo / float(total_demo)

    print("Demo result")
    print("  Training accu: " + str(setdigit(digits).format(train_accu_rec))  + " %")  
    print("  Testing  accu: " + str(setdigit(digits).format(test_accu_rec))   + " %")
    print("  Valid.   accu: " + str(setdigit(digits).format(demo_accu_rec))   + " %")

    sns.heatmap(cf_train)
    plt.title("confusion mtx. training")
    plt.xlabel("predictions")
    plt.ylabel("labels")
    plt.show()
    
    sns.heatmap(cf_test)
    plt.title("confusion mtx. testing")
    plt.xlabel("predictions")
    plt.ylabel("labels")
    plt.show()
    
    sns.heatmap(cf_demo)
    plt.title("confusion mtx. validation")
    plt.xlabel("predictions")
    plt.ylabel("labels")
    plt.show()

    return 0
    
# construct model
model = CNN()
print(stat(model, (3, 40, 40)))

model = torch.load(model_path)
model = model.to(device)

loss_func = nn.CrossEntropyLoss()


Demo(model, loss_func,  input_shape, trainloader, testloader, demoloader)


dummy_input  = torch.randn(1, 3, 40, 40)

# Export the model in onnx format
export = input("export this version to onnx? [y/n] ")
if (export == 'y'):
    print("exporting model...")
    model.eval()
    torch.onnx.export(model, dummy_input,
                      "model.onnx",
                      input_names=["input"],
                      output_names = ["output"],
                      opset_version=12,
                      dynamic_axes={"input": {0:"batch_size"},
                                    "output":{0:"batch_size"}})
    print("model exported")
    onnx_model = onnx.load("model.onnx")
    onnx.checker.check_model(onnx_model)