#!/usr/bin/env python
#tensorflow 2.0
# @Time    : 2019/12/8 15:56
# @File    : f4.py
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

image_value=tf.io.read_file("t1.jpg")
img=tf.image.decode_jpeg(image_value,channels=3)
print(type(image_value))  #<class 'tensorflow.python.framework.ops.EagerTensor'>
print(type(img))# <class 'tensorflow.python.framework.ops.EagerTensor'>

print(type(img.numpy()))
print(img.numpy().shape)
print(img.numpy().dtype)

plt.figure(1)
plt.imshow(img.numpy())
plt.show()