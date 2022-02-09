import tensorflow as tf
import numpy as np
import time

# traing data 
x_data = [[1,2,1], [1,3,2], [1,3,4], [1,5,5], [1,7,5], [1,2,5], [1,6,6], [1,7,7]]
y_data = [[0,0,1], [0,0,1], [0,0,1], [0,1,0], [0,1,0], [0,1,0], [1,0,0], [1,0,0]]

# test data
x_test = [[2,1,1], [3,1,2], [3,3,4]]
y_test = [[0,0,1], [0,0,1], [0,0,1]]

# learning_rate = 0.1 
# bing learning rate
# learning_rate = 1.5  이 두 경우에서 안되는 이유는 keras의 optimizer에서 이러한 rate 경우를 다 처리해버리기 때문?
# learning_rate = 10.0

# small learning rate
# learning_rate = 1e-10 learning rate의 값이 너무 작기 때문에 정확도 0 뜸. 멈춰있기 때문


tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=3, input_dim=3, activation="softmax"))

tf.model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.SGD(learning_rate), metrics=['accuracy'])
tf.model.summary()

history = tf.model.fit(x_data, y_data, epochs=200)

# Predict | test data 값을 바탕으로 예측값을 게산할 수 있다.
x_predict = tf.model.predict(np.array(x_test))
prediction = np.argmax(x_predict, axis=1)
print("Prediction: ", prediction)

accuracy = tf.model.evaluate(x_test, y_test)[1]
# Accuracy | test data 값을 바탕으로 정확도를 계산할 수 있다.
print("Accuracy: ", accuracy)

