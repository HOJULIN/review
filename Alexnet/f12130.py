#!/usr/bin/env python
#tensorflow 2.0
# @Time    : 2019/12/13 17:03
# @File    : f12130.py
import tensorflow as tf
def fun(x):
    return x**2

ds=tf.data.Dataset.range(5)
for i in ds:
    print(i)
ds=ds.map(fun)
print("------")
print(ds)
for i in ds:
    print(i)
print(1)

# tf.Tensor(0, shape=(), dtype=int64)
# tf.Tensor(1, shape=(), dtype=int64)
# tf.Tensor(2, shape=(), dtype=int64)
# tf.Tensor(3, shape=(), dtype=int64)
# tf.Tensor(4, shape=(), dtype=int64)
# ------
# <MapDataset shapes: (), types: tf.int64>
# tf.Tensor(0, shape=(), dtype=int64)
# tf.Tensor(1, shape=(), dtype=int64)
# tf.Tensor(4, shape=(), dtype=int64)
# tf.Tensor(9, shape=(), dtype=int64)
# tf.Tensor(16, shape=(), dtype=int64)