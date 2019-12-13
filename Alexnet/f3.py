#!/usr/bin/env python
#tensorflow 2.0
# @Time    : 2019/12/8 13:44
# @File    : f3.py
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
print("jj")


# if not tf.gfile.exists(DATA_DIRECTORY):
#         tf.gfile.makedirs(DATA_DIRECTORY)
# with tf.gfile.GFile(filepath) as f:
image_raw=tf.io.gfile.GFile('t3.jpg','rb').read()
# print(type(image_raw))  #<class 'bytes'>
img=tf.image.decode_jpeg(image_raw,channels=3)
# img2=tf.image.rgb_to_grayscale(img)

print(type(img))  #<class 'tensorflow.python.framework.ops.EagerTensor'>


print(type(img.numpy()))  #<class 'numpy.ndarray'>
print(img.numpy().shape)  #(493, 690, 3)  (高度，宽度，通道数)
print(img.numpy().dtype)  #uint8
print(img.numpy().shape)
plt.figure(1)
plt.imshow(img.numpy())
# plt.figure(2)  设置为灰度图像  cmap="gray"
# plt.imshow(img2.numpy().squeeze(),cmap="gray")
plt.show()

# -------------------------------
# <class 'tensorflow.python.framework.ops.EagerTensor'>
# <class 'numpy.ndarray'>
# (493, 690, 3)
# uint8