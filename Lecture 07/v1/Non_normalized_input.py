import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# traing data 
tr_data = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973], 
                [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
                [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
                [816, 820.958984, 1008100, 815.48999, 819.23999], 
                [819.359985, 823, 1188100, 818.4699711, 818.97998], 
                [819, 823, 1198100, 816, 820.450012], 
                [811.700012, 815.25, 1098100, 809.780029, 813.669983], 
                [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

# 데이터 값이 갑작스럽게 커지는 경우가 발생할 때 
# MinMaxScaler를 이용해 0과 1사이의 값으로 출력되게함

xy = MinMaxScaler().fit_transform(tr_data)
print(xy)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# test data
x_test = [[2,1,1], [3,1,2], [3,3,4]]
y_test = [[0,0,1], [0,0,1], [0,0,1]]

X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.random_normal([4, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(-tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                            feed_dict={X: x_data, Y: y_data})
        print(step, "Cost : ", cost_val, "\nPrediction : \n", hy_val)