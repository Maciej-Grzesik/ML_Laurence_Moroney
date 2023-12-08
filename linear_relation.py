import numpy as np
from keras import Sequential
from keras.layers import Dense

l0 = Dense(units=1, input_shape=[1]) # layer with one neuron
model = Sequential([l0]) # defining layers
model.compile(optimizer='sgd', loss='mean_squared_error') # choosing optimizer and loss function

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 0.0, 2.0, 4.0, 6.0, 8.0], dtype=float)

model.fit(xs, ys, epochs=5000) # fitting xs to ys

print(model.predict([10.0]))
print(l0.get_weights())
