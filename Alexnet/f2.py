#!/usr/bin/env python
#tensorflow 2.0
# @Time    : 2019/12/7 18:56
# @File    : f2.py
import numpy as np
import cv2
import matplotlib.pyplot as plt


import tensorflow as tf
a=tf.Variable([[-1,0],[1,21]])

# b=tf.cast(a,tf.bool)
print(a)
print(a.name)
print(a.trainable)