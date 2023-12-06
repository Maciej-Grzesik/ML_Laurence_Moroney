import tensorflow as tf
import numpy as np
from keras import Sequential
from keras.layers import Dense

l0 = Dense(units=1, input_shape=[1])
model = Sequential([l0])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 0.0, 2.0, 4.0, 6.0, 8.0], dtype=float)

model.fit(xs, ys, epochs=5000)

print(model.predict([10.0]))
print(l0.get_weights())
