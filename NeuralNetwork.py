import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
def NeuralNetwork(train_data,test_data):
    X_train = train_data[0].astype(np.float32)
    X_test = test_data[0].astype(np.float32)
    y_train = train_data[1]
    y_test= test_data[1]
    for i in range(len(X_train[0, :])):#make all columns to range 0-1
        if max(X_train[:, i]) != 0:
            X_train[:, i] = X_train[:, i] / max(X_train[:, i])
        if max(X_test[:, i]) != 0:
            X_test[:, i] = X_test[:, i] / max(X_test[:, i])
    model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(45, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(max(test_data[1])+1)
    ])
    model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
