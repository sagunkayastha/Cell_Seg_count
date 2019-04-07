
import matplotlib.pyplot as plt
import cv2
import numpy as np
A = np.load('A_.npy')


X = np.load('../bb_cell/train_img.npy')
y = np.load('../bb_cell/train_mask.npy').astype(np.uint8) *255



cv2.imshow('f',X[100])
cv2.imshow('f1',A[100])
cv2.imshow('f2', y[100])


cv2.waitKey(0)
