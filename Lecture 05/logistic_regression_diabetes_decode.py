import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

filename_queue = tf.train.string_input_producer(['data-03-diabetes.csv'], shuffle=False, name="filename_queue")

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults = [[0.]]*9
xy = tf.decode_csv(value, record_defaults=record_defaults)

train_x_batch, train_y_batch = \
    tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([8, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# Hypothesis using sigmoid : tf.div(1., 1. + tf.exp(tf.matmul(X, W) + b))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(10001):
        x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                            feed_dict={X: x_batch, Y: y_batch})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict = {X: x_batch, Y: y_batch}))
    coord.request_stop_()
    coord.join(threads)
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_batch, Y: y_batch})
    print("\nHypothesis : ", h, "\nCorrect (Y) : ", c, "\nAccuracy : ", a)
