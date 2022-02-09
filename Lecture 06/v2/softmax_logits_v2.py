import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

nb_classes = 7      # 7개의 클래스가 있다

# y_data one hot encoding
y_one_hot = tf.keras.utils.to_categorical(y_data, nb_classes)
print("one_hot : ", y_one_hot)

tf.model = tf.keras.Sequential()

tf.model.add(tf.keras.layers.Dense(units=nb_classes, input_dim=16, activation='softmax'))

tf.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.1), metrics=['accuracy'])
tf.model.summary()

history = tf.model.fit(x_data, y_one_hot, epochs=1000)

data = np.array([[0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0]])
test_data = tf.model.predict(data)
print(test_data, np.argmax(test_data, axis=1))

data = np.array(x_data)
test_data = tf.model.predict(data)
pred = np.argmax(test_data, axis=1)

for p,y in zip(pred, y_data.flatten()):         # flatten : [[1], [0]] -> [ 1, 0 ]
    print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))