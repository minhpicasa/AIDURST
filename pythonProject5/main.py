import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd

# Load the Excel file
file_path = 'Data.xlsx'
df = pd.read_excel(file_path)
X = df.iloc[:, :-1]  # All rows, all columns except the last one
y = df.iloc[:, -1]   # All rows, only the last column
X = X.to_numpy()
y= y.to_numpy()

print("shape of X" + str(X.shape))
print("shape of Y" + str(y.shape))

model = Sequential(
    [
        tf.keras.Input(shape=(14,)),
        tf.keras.layers.Dense(25, activation="relu"),
        tf.keras.layers.Dense(15, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ], name = "my_model"
)

print(model.summary())

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.0098),
)

model.fit(
    X,y,
    epochs=200,
    batch_size=16
)


prediction = model.predict(X[19].reshape(1,14))  # a one
print(f" predicting a one: {prediction}")


prediction1 = model.predict(X[20].reshape(1,14))  # a one
print(f" predicting a one: {prediction1}")


prediction2 = model.predict(X[21].reshape(1,14))  # a one
print(f" predicting a one: {prediction2}")
#if prediction >= 0.5:
#    yhat = 1
#else:
#    yhat = 0
#print(f"prediction after threshold: {yhat}")
