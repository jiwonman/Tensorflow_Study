import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name="weight")           #random_normal()  값 안에 shape을 줌
b = tf.Variable(tf.random_normal([1]), name="bias")

# Our hypothesis XW+b 
hypothesis = x_train * W + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))      #reduce_mean() 함수 안의 값을 평균내주는 것 cost함수에서 1/m시그마 부분과 동일

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)            #train은 그래프 상의 하나의 노드 이름 

# Launch the graph in a session
sess = tf.Session()

# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))

# train을 실행시킴으로써 아래 cost 아래 hypothesis 아래 w와 b를 알 수 있다.
# 따라서, optimizer를 실행시키면 cost값은 적은 값으로 수렴, W값은 1로 수렴, b값은 0으로 수렴