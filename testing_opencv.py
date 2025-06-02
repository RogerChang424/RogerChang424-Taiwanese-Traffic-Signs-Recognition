# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:11:23 2024

@author: user
"""

import cv2
import numpy as np
from scipy import special
import time

import matplotlib.pyplot as plt
import seaborn as sns

# precision for time duration
def setdigit(x):
    return "{:10." + str(x) + "f}"

# class indices to directory indices
lookup = np.array([0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 3, 4, 5, 6, 7, 8, 9])
net = cv2.dnn.readNetFromONNX("model.onnx")

width  = 40

img_quantity       = 7 * 27
print_false_pred   = True
confid_criterion   = 90



print()
conf = np.zeros([28, 28])
correct = 0
above_confid = 0
above_confid_correct = 0

time_start = time.time()
for idxC in range (28):
    for idxQ in range(7):
        """
        if(idxQ % 10 < 2):
            continue
        """
        image = cv2.imread("./balanced_val/(" + str(idxC) + ")/(" + str(idxQ + 280) + ").png").astype(np.float32)
        image /= 255.0
        image *= 2.0
        image -= 1.0

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        blob  = cv2.dnn.blobFromImage(image)
        net.setInput(blob)
        output = special.softmax(net.forward())
        pred = np.argmax(output)
        pred_lookup = lookup[pred]
        conf[idxC, pred_lookup] += 1
        #print(output)
        if(pred_lookup == idxC):
            correct += 1
        if(output[0, pred] > confid_criterion/100.0):
            above_confid += 1
            if(pred_lookup == idxC):
               above_confid_correct += 1
        
time_end = time.time()
dur = time_end - time_start



print("corrects = " + str(correct))
print("accu = " + str(setdigit(4).format(correct * 100/img_quantity)) + "%")
print("corrects(above confidence criterion) = " + str(above_confid_correct))
print("accu(above confidence criterion)     = " + str(setdigit(4).format(above_confid_correct * 100/above_confid)) + "%")
print("total(above confidence criterion)    = " + str(above_confid))           
print("total duration: " + str(setdigit(6).format(dur)) + " secs")
print("avg.  duration: " + str(setdigit(6).format(dur/img_quantity)) + " secs/img")
print("avg.     speed: " + str(setdigit(2).format(img_quantity/dur)) + " imgs/sec")
print()

sns.heatmap(conf)
plt.xlabel("prediction")
plt.ylabel("label")
plt.title("confusion matrix")
plt.show()
cv2.destroyAllWindows()