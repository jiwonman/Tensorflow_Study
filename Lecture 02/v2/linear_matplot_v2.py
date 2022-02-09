import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

X = [1,2,3]
Y = [1,2,3]

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=1))

sgd = tf.keras.optimizers.SGD(lr=0.01) 
tf.model.compile(loss='mse', optimizer=sgd)

tf.model.summary()

history = tf.model.fit(X, Y, epochs=100)

y_predict = tf.model.predict(np.array([5, 4]))
print(history.params)
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()