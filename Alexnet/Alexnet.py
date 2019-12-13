#!/usr/bin/env python
#tensorflow 2.0
# @Time    : 2019/11/23 20:57
# @File    : Alexnet.py
#  结构如图https://www.jianshu.com/p/00a53eb5f4b3

import tensorflow as tf
try:
    import tensorflow.python.keras as keras
except:
    import tensorflow.keras as keras
# from tensorflow.python.keras import models
# from tf.keras import layers
try:
    from tensorflow.python.keras import layers
except:
    from tensorflow.keras import layers

#注意要加上python

def AlexNet_inference(in_shape):    #输入图片的形状
    model=keras.models.Sequential(name="AlexNet")
    model.add(layers.Conv2D(96,(11,11),strides=(2,2),input_shape=(in_shape[1],in_shape[2],in_shape[3]),
                     padding='same',activation='relu',))
    #kernel_initializer='uniform'这个是用来神魔？
    model.add(layers.MaxPooling2D( pool_size=(3, 3), strides=(2, 2)))
    model.add(layers.Conv2D(256,(5,5),strides=(1,1),
                     padding='same',activation='relu'))
    model.add(layers.MaxPooling2D( pool_size=(3, 3), strides=(2, 2)))
    model.add(layers.Conv2D(384,(3,3),strides=(1,1),
                     padding='same',activation='relu'))
    model.add(layers.Conv2D(384, (3, 3), strides=(1, 1),
                     padding='same', activation='relu' ))
    model.add(layers.Conv2D(256, (3,3), strides=(1, 1),
                            padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(2048,activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2048,activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',  # 不能直接用函数，否则在与测试加载模型不成功！
                  metrics=['accuracy'])
    model.summary()
    return model







