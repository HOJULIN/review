#!/usr/bin/env python
#tensorflow 2.0
# @Time    : 2019/12/13 11:05
# @File    : f10.py
import  tensorflow as tf
var=tf.one_hot(indices=[1,2,3],depth=4,axis=0,dtype=tf.int64)
print(var)