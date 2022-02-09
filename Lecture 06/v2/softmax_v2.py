import tensorflow as tf
import numpy as np

x_raw = [[1,2,1,1], [2,1,3,2], [3,1,3,4], [4,1,5,5], [1,7,5,5], [1,2,5,6], [1,6,6,6], [1,7,7,7]]
y_raw = [[0,0,1], [0,0,1], [0,0,1], [0,1,0], [0,1,0], [0,1,0], [1,0,0], [1,0,0]]

x_data = np.array(x_raw, dtype=np.float32)
y_data = np.array(y_raw, dtype=np.float32)

nb_classes = 3

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=nb_classes, input_dim=4))

tf.model.add(tf.keras.layers.Activation('softmax'))

tf.model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.SGD(lr=0.1), metrics=['accuracy'])
tf.model.summary()

history = tf.model.fit(x_data, y_data, epochs=2000)

# 1개씩 일일히 계산 하는 법
a = tf.model.predict(np.array([[1, 11, 7, 9]]))
print(a, tf.keras.backend.eval(tf.argmax(a, axis=1)))

b = tf.model.predict(np.array([[1, 3, 4, 3]]))
print(b, tf.keras.backend.eval(tf.argmax(b, axis=1)))

# or use argmax embedded method, predict_classes
# predict_classes 코드가 없어져서 np.argmax사용하여 직접처리
c = tf.model.predict(np.array([[1, 1, 0, 1]]))
c_onehot = np.argmax(c, axis=1)
print(c, c_onehot)

#
all = tf.model.predict(np.array([[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]))
all_onehot = np.argmax(all, axis=1)
print(all, all_onehot)
