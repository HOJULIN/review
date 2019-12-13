#!/usr/bin/env python
#tensorflow 2.0
# @Time    : 2019/12/9 10:27
# @File    : f7.py

import os
import numpy  as np
import IPython.display as display

import  matplotlib.pyplot as plt

import tensorflow as tf
#制作tfrecord文件
image_raw_data_jpg2= tf.io.gfile.GFile(r'C:\Users\Administrator\Desktop\test\cat\c2.jpg','rb').read()
from PIL import Image
cwd=os.getcwd()
print(cwd)
classes={'cat','dog'}
writer=tf.io.TFRecordWriter("train_own.tfrecords")  #设置要生成的文件
# file_list=os.listdir(r'C:\Users\Administrator\Desktop\test\dog')
# print(file_list)

# file_list=[r'C:\Users\Administrator\Desktop\test\dog\',
# r'C:\Users\Administrator\Desktop\test\cat']
for index,name in enumerate(classes):
    # 构建路径名的时候，注意"/" "\"的使用
    # 在Linux中以"/"分割，在Windows下以"\"分割，各位根据自己情况修改
    class_path=r'C:\Users\Administrator\Desktop\test'+'\\'+name+'\\'
    # print(os.listdir(class_path))

    for img_name in os.listdir(class_path):
            img_path=class_path+img_name
            # print(img_path)
            img=tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img,channels=3) #此处img是tf.Tensor格式Eager Tensor格式，需要转化为numpy
            # print(img.shape)
            exam = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        # 'name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[name.encode('utf-8')])),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                        #'shape': tf.train.Feature(
                            #int64_list=tf.train.Int64List(value=[img.shape[0], img.shape[1], img.shape[2]])),
                        'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(img.numpy())]))
                    }
                )
            )
            writer.write(exam.SerializeToString())
writer.close()
print(1)

#make_ftrecord.py”的文件

#读取一个make_ftrecord.py
dataset = tf.data.TFRecordDataset("train_own.tfrecords")  # 打开一个TFrecord

feature_description = {
    #'name': tf.io.FixedLenFeature([], tf.string, default_value='0.0'),
    'label': tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    #'shape': tf.io.FixedLenFeature([3], tf.int64),
    'data': tf.io.FixedLenFeature([], tf.string)
}


def _parse_function(exam_proto):  # 映射函数，用于解析一条example
    return tf.io.parse_single_example(exam_proto, feature_description)


dataset= dataset.repeat (1) # 读取数据的重复次数为：1次，这个相当于epoch
dataset = dataset.shuffle (buffer_size = 2000) # 在缓冲区中随机打乱数据
dataset= dataset.map (_parse_function) # 解析数据
dataset = dataset.batch (batch_size = 1) # 每10条数据为一个batch，生成一个新的Dataset
# dataset= dataset.map (_parse_function) # 解析数据
# train_iterator = dataset.make_one_shot_iterator()




for raw_record in dataset.take(1):             #raw_record是字典，有['data']是tf.tensor,dtype=string
    print(raw_record['label'])
    example=tf.train.Example()
    print(example)
    print('>>>>>>>>>>>>>>>>>>>>>>>>')
    example.ParseFromString(raw_record.numpy())
    print(example)
    print(1)


for image in dataset:
    image_raw=image['data'].numpy()
    plt.figure(1)
    plt.imshow(image_raw)
    plt.show()
    display.display(display.Image(data=image_raw))  #只能在juptyer上显示图片

    # plt.show()
#print(dataset)
train_images, train_labels = dataset
print(type(train_images),type(train_labels))
print(train_images)
print('///////////////////')
print(train_labels)




for row in dataset.take(5):
    print(row['label'])
    # plt.figure()
    # plt.imshow(np.frombuffer(row['data'].numpy(),dtype=np.uint8))
    # plt.show()
    # print(2)

    # print(np.frombuffer(row['data'].numpy(),dtype=np.uint8))


shape = []
batch_data_x, batch_data_y = np.array([]), np.array([])
for item in batch.take(1): # 测试，只取1个batch
    shape = item['shape'][0].numpy()
    for data in item['data']: # 一个item就是一个batch
        img_data = np.frombuffer(data.numpy(), dtype=np.uint8)
        batch_data_x = np.append (batch_data_x, img_data)
    for label in item ['label']:
        batch_data_y = np.append (batch_data_y, label.numpy())

batch_data_x = batch_data_x.reshape ([-1, shape[0], shape[1], shape[2]])
print (batch_data_x.shape, batch_data_y.shape) # = (10, 480, 640, 3) (10,)
# 我的图片数据时480*640*3的





