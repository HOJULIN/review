#!/usr/bin/env python
#tensorflow 2.0
# @Time    : 2019/12/8 20:22
# @File    : f6.py
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import os

class BatchRename():
    '''
    批量重命名文件夹中的图片文件

    '''
    def __init__(self):
        self.path = r'C:\Users\Administrator\Desktop\1'     #图片所在的目录

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        print("总的图片数量"+str(total_num))
        i = 0
        for item in filelist:
            # print(filelist)
            if item.endswith(('.jpg','.png','.jfif')):

                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), "ip"+str(i) + '.jpg')
                try:
                    os.rename(src, dst)
                    i = i + 1
                except:
                    # print(item)
                    continue


if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()
