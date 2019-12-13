#!/usr/bin/env python
# tensorflow 2.0
# @Time    : 2019/11/24 13:25
# @File    : Alexbet-train.py

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
import  matplotlib.pyplot as plt
print(tf.__version__)

mnist=tf.keras.datasets.mnist
import Alexnet


(x_train,y_train),(x_test,y_test)=mnist.load_data()
#x_train的形状为(60000, 28, 28)，y_train的形状为(60000,),x_test的形状为(10000, 28, 28)


print(y_train)
x_train=x_train.reshape((-1,28,28,1))
#此型状为(60000, 28, 28, 1) 60000是数量，1代表是单色图像
x_test=x_test.reshape((-1,28,28,1))

#print(x_train.shape)
print(x_train.shape[0],x_train.shape[1])
x_shape=x_train.shape

def AlexNet_train():
    AlexNet_model=Alexnet.AlexNet_inference(x_shape)
    totall_epochs=0
    epochs=10

    while(True):
        history=AlexNet_model.fit(x_train,y_train,batch_size=64,epochs=epochs,validation_split=0.1)

        #fit函数返回一个History的对象，其History.history属性记录了损失函数和其他指标的数值随epoch变化的情况
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        #验证集的准确性
        plt.legend(['training','valivation'],loc='upper left')
        plt.show()

        res=AlexNet_model.evaluate(x_test,y_test)
        print(res)

        totall_epochs+=epochs
        model_save_dir='AlexNet_model_'+str(totall_epochs)+".h5"
        AlexNet_model.save(model_save_dir)

        keyVal=input("0:quit,1:continue")
        keyVal=int(keyVal)
        if 0==keyVal:
            break
        else:
            epochs=keyVal


AlexNet_train()


