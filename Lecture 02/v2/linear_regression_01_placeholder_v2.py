import numpy as np
import tensorflow as tf

x_train = [1, 2, 3, 4, 5]
y_train = [2.1, 3.1, 4.1, 5.1, 6.1]

tf.model = tf.keras.Sequential()
# units == output shape, input_dim == input shape
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=1))

sgd = tf.keras.optimizers.SGD(lr=0.01)  # SGD == standard gradient descendent, lr == learning rate
tf.model.compile(loss='mse', optimizer=sgd)  # mse == mean_squared_error, 1/m * sig (y'-y)^2

# prints summary of the model to the terminal
tf.model.summary()

# fit() executes training
tf.model.fit(x_train, y_train, epochs=200)

# predict() returns predicted value
print(tf.model.predict(np.array([5])))
print(tf.model.predict(np.array([2.5])))
print(tf.model.predict(np.array([1.5, 3.5])))