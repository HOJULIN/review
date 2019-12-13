#!/usr/bin/env python
#tensorflow 2.0
# @Time    : 2019/12/11 21:22
# @File    : f8.py
import os
import cv2
import os
import numpy  as np
import IPython.display as display

import  matplotlib.pyplot as plt
img=cv2.imread('t1.jpg')
        # print(filename)
img2=cv2.resize(img,(224,224))

plt.figure(num=1)
plt.imshow(img)
plt.figure(num=2)

plt.imshow(img2)
plt.show()