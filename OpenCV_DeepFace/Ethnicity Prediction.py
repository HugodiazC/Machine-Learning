# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 15:53:48 2021

@author: L01506162
"""


from deepface import DeepFace
import cv2 
import matplotlib.pyplot as plt

img = cv2.imread("img1.jpg")
plt.imshow(img[:,:,::-1])
plt.show()
