#!/usr/bin/env python
#tensorflow 2.0
# @Time    : 2019/11/24 18:45
# @File    : f1.py.py

import  numpy as np

import tensorflow as tf

# print(tf.__version__)  np.linalg.inv(c)
# print(tf.test.is_gpu_available())
data=[]
for i in range(100):
    x=np.random.uniform(-10,10)
    eps=np.random.normal(0,0.1)
    y=1.477*x+0.089+eps
    data.append([x,y])
    #print(data)


data=np.array(data)
print(type(len(data)))
# print(  type(data[0][0]),
#         data[0][1])
def mse(b,w,points):
    totalError=0
    print(len(points))
    for i in range(0, len(points)):
        x=points[i][0]
        y=points[i][1]
        totalError+=(y-(w*x+b))**2
    return totalError/float(len(points))

def step_gradient(b_current,w_current,points,lr):
    b_gradient=0
    w_gradient=0
    M=float(len(points))
    for i in range(0,  len(points)):
        x=points[i][0]
        y=points[i][1]
        b_gradient+=(2/M)*(w_current*x+b_current-y)
        w_gradient+=(2/M)*x*(w_current*x+b_current-y)
    new_b=b_current-(lr*b_gradient)
    new_w=w_current-(lr*w_gradient)

    return [new_b,new_w]

def gradient_descent(points,starting_b,starting_w,lr,num_iterations):
    b=starting_b
    w=starting_w
    for step in range(num_iterations):
        b,w=step_gradient(b,w,points,lr)
        loss=mse(b,w,points)
        if step%50==0:
            print(f"iteration:{step},loss:{loss}")
    return [b,w]


def main():
    lr=0.01
    initial_b=0.1
    initial_w=0.1
    num_iterations=1000
    [b,w]=gradient_descent(data,initial_b,initial_w,lr,num_iterations)
    loss=mse(b,w,data)
    print(print(f"final lossloss:{loss},w:{w},b:{b}"))

main()




