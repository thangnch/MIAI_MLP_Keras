
# Imports
import keras
#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical

#Thai import them vao
import numpy as np
import pandas as pd
import sys

#Train/test split
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

# Configuration options
feature_vector_length = 10 #tuong ung voi 10 gia tri cua 10 cam bien
num_classes = 52535 # chieu dai cua tap du lieu

# Load the data
def read_data():
    dataset = pd.read_csv('sensordata1.csv')
    X = dataset.iloc[:,1:11].values
    y = dataset.iloc[:,-1:].values
    return X,y

X,y = read_data()

# Convert y to one-hot
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(y)
y= enc.transform(y).toarray()
print ("X_size:", X.shape,"Y_size:", y.shape)
#sys.exit()

#train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print ("X_train size is:", X_train.shape,"y_train size is:", y_train.shape)
print ("X_test size is:", X_test.shape,"y_test size is:", y_test.shape)
#
in_dim = X.shape[1]
out_dim = y.shape[1]
# print ("input_dimension:", in_dim, "output_dimension:", out_dim)
#
# Create the model
model = Sequential()
model.add(Dense(512, input_dim=10, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(35, activation = "softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=256, verbose=1, validation_split=0.2)
model.save("model.h5")
#
# # Test the model after training
test_results = model.evaluate(X_test, y_test, verbose=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')