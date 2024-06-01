import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess data
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Define model architecture
model = Sequential()
model.add(Flatten(input_shape=(28, 28, 1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=10, batch_size=100, validation_data=(X_test, y_test))

# Save model
model.save('my_model.h5')

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print('Accuracy Testing MLP:', accuracy)

# Visualize model evaluation
epochs = range(10)
losses = history.history['loss']
val_losses = history.history['val_loss']
plt.plot(epochs, losses, 'r', label='Training Loss ANN')
plt.plot(epochs, val_losses, 'b', label='Validation Loss ANN')
plt.legend()
plt.show()

# Load model and make predictions
loaded_model = keras.models.load_model('my_model.h5')
predictions = loaded_model.predict(X_test)
print('Actual Label:', np.argmax(y_test[30]))
print('Predicted Label:', np.argmax(predictions[30]))
