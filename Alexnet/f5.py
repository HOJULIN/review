#!/usr/bin/env python
#tensorflow 2.0
# @Time    : 2019/12/8 17:17
# @File    : f5.py
import os
import cv2
#jpeg后缀的图像，其由于jpeg图像本身的编解码问题，写入时的编码与读取时的解码所得不能完美互为逆操作，从而导致的每次写入之后，读取的值都不同。
#写到这里，搜了搜关键字”jpeg编解码 有损“的结果 ，也确实如此。Jpeg是一种有损压缩，而png是无损压缩。
#到这里，也就解决了这个其实算不上bug的bug：

for i,filename in enumerate(os.listdir(r'C:\Users\Administrator\Desktop\t')):

    if(filename.endswith('.jpg'or '.png'or'.jfif')):

        filename.endswith()

        img=cv2.imread(filename)
        # print(filename)
        cv2.imwrite(str(i)+".png",img)
    else:
        break
