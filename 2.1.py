import pandas as pd
from keras.src.layers import Dense

path_to_csv = "C:\\Users\\rasvitha\\OneDrive\\NN&DeepLearning_Lesson7_SourceCode\\diabetes.csv"
import keras
import pandas
from keras.models import Sequential

# load dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

dataset = pd.read_csv("C:\\Users\\rasvitha\\OneDrive\\NN&DeepLearning_Lesson7_SourceCode\\diabetes.csv", header=None).values

X_train, X_test, Y_train, Y_test = train_test_split(dataset[:, 0:8], dataset[:, 8], test_size=0.25, random_state=87)
np.random.seed(155)
my_first_nn = Sequential()  # create model
my_first_nn.add(Dense(20, input_dim=8, activation='relu'))  # hidden layer
my_first_nn.add(Dense(4, activation='relu'))  # hidden layer
my_first_nn.add(Dense(1, activation='sigmoid'))  # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100,
                                     initial_epoch=0)
print(my_first_nn.summary())
print(my_first_nn.evaluate(X_test, Y_test))
