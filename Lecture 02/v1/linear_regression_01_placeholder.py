from turtle import shape
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

W = tf.Variable(tf.random_normal([1]), name="weight")           #random_normal()  값 안에 shape을 줌
b = tf.Variable(tf.random_normal([1]), name="bias")
X = tf.placeholder(tf.float32, shape=[None])                    #placeholder를 쓰는 가장 좋은 이유는 우리가 직접 모의 값들을 넣을 수 있다는 것이다.(feed_dict 안에)
Y = tf.placeholder(tf.float32, shape=[None])

# Our hypothesis XW+b 
hypothesis = X * W + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis -Y))      #reduce_mean() 함수 안의 값을 평균내주는 것 cost함수에서 1/m시그마 부분과 동일

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)            #train은 그래프 상의 하나의 노드 이름 

# Launch the graph in a session
sess = tf.Session()

# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, train], feed_dict={X: [1,2,3,4,5], Y: [2.1, 3.1, 4.1, 5.1, 6.1]})
    if step % 20 == 0:
            print(step, cost_val, W_val, b_val)

# Testing our model
print(sess.run(hypothesis, feed_dict={X: [5]}))
print(sess.run(hypothesis, feed_dict={X: [2.5]}))
print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))