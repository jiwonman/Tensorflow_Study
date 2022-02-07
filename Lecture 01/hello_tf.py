import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

hello = tf.constant("Hello, TensorFlow!")

sess = tf.Session()

print(sess.run(hello))

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

print("node1 : ", node1, "node2 : ", node2)
print("node3 : ", node3)

sess = tf.Session()
print("sess.run(node1, node2) : ", sess.run([node1, node2]))
print("sess.run(node3) : ", sess.run(node3))

a = tf.placeholder(tf.float32)          #placeholder라는 노드를 만들 수 있다. (넘겨주는 행위)
b = tf.placeholder(tf.float32)
adder_node = a + b

print(sess.run(adder_node, feed_dict = {a: 3, b: 4.5}))         #seesion을 통해 그래프를 실행 시킬 때 feed_dict로 값을 넘겨준다.
print(sess.run(adder_node, feed_dict = {a: [1,3], b: [2,4]}))

