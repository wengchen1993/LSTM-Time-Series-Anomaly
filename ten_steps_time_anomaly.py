# Import libraries
import glob
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from keras.regularizers import L1L2
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, LSTM, Dropout, Activation, TimeDistributed, Lambda
from keras.optimizers import Adam
from keras.utils import np_utils

def load_data():
    # Load relevant datas
    df = pd.read_csv("INSERT INPUT FILE DIR")
    input_data = df['column'].tolist()
    input_data = [(idata/10000) for idata in input_data]
    return input_data

def create_train_test(sc, input_data, look_back):

    # Split datasets into 4/5 training and 1/5 test sets
    input_data_length = len(input_data)
    train_test_boundary = int(input_data_length * 4 / 5) + 1

    # Create train datasets and scale it
    train_data = input_data[:train_test_boundary]
    train_data = [[elem] for elem in train_data]
    train_data_scaled = sc.fit_transform(train_data)

    train_data_length = len(train_data)
    x_train = []
    y_train = []
    for i in range(look_back, train_data_length - 9):
        x_train.append(train_data_scaled[i - look_back : i, 0])
        y_train.append(train_data_scaled[i + 9, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    temp_input_data = [[elem] for elem in input_data]
    input_data_scaled = sc.fit_transform(temp_input_data)

    x_test = []
    for j in range(train_test_boundary, input_data_length - 9):
        x_test.append(input_data_scaled[j - look_back : j, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    y_test = input_data[train_test_boundary + 9:]

    return (x_train, y_train), (x_test, y_test)

def build_model(look_back):
    # Building models
    model = Sequential()
    model.add(LSTM(units = 3, return_sequences = True, input_shape = (None,1)))
    model.add(LSTM(units = 3))
    model.add(Dense(units = 3))
    model.add(Dense(units = 1))
    adam = Adam(lr = 0.0005)
    model.compile(optimizer = adam, loss = 'mean_squared_error')
    print(model.summary())

    return model

def draw_plot(y_test, results):
    # Visualising the results
    plt.plot(y_test[-500:], color = 'red', label = 'Real Data Projection')
    plt.plot(results[-500:], color = 'blue', label = 'Predicted Data Projection')
    plt.title('cephi042_bytes_rx')
    plt.xlabel('Time')
    plt.ylabel('Bytes')
    plt.legend()
    plt.show()

def print_stats(actual_res, predict_res):
    rmse = math.sqrt(mean_squared_error(actual_res, predict_res))
    mse = mean_squared_error(actual_res, predict_res)
    mae = mean_absolute_error(actual_res, predict_res)
    mae_threshold = np.mean(mae) + np.std(mae)

    print("Actual MAE is {}".format(mae))
    print("Actual MAE threshold is {}".format(mae_threshold))
    print("Actual MSE is {}".format(mse))
    print("Actual RMSE is {}".format(rmse))

    avg_curr = np.mean(input_data)
    print("MAE is {} %".format( (mae/avg_curr) * 100 ))
    print("MSE is {} %".format( (mse/avg_curr) * 100 ))
    print("RMSE is {} %".format( (rmse/avg_curr) * 100 ))
    print("Average :{}".format(avg_curr))

def save_weights_layout(model):
    model_json = model.to_json()
    with open("./weights_rx_10.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("weights_rx_10.h5")


if __name__ == "__main__":
    # Declare variables
    lag_steps = 14   # Using [(t-14), (t-13) ... (t-1)] to predict t
    bsize = 128      # standard batch size for every weight update
    iterations = 50  # Number of epochs, usually in the range of 50 - 1000, making it 5 for testing purpose

    # Regularization for overfit circumstances
    #reg = [L1L2(l1=0.0, l2=0.0), L1L2(l1=0.01, l2=0.0), L1L2(l1=0.0, l2=0.01), L1L2(l1=0.01, l2=0.01)]

    # Scaler for scaling data from arbitrary range into 0 - 1
    sc = MinMaxScaler(feature_range = (0, 1))

    # Start Measuring computation time
    start = time.time()

    # Loading input data from folders
    input_data = load_data()

    # Split data into train and test sets
    train_sets, test_sets =  create_train_test(sc, input_data, lag_steps)

    x_train, y_train = train_sets[0], train_sets[1]
    x_test, y_test = test_sets[0], test_sets[1]

    # Build LSTM model
    model = build_model(lag_steps)

    # Training train_datasets
    model.fit(x_train, y_train, epochs = iterations, batch_size = bsize, validation_split = 0.2)

    # Generate predicted resutlts
    y_predicted = model.predict(x_test)
    y_predicted = sc.inverse_transform(y_predicted)
    y_predicted = list(y_predicted.flatten())
    end = time.time()

    # Save weights and model
    # save_weights_layout(model)

    # Printing some statistics
    print_stats(y_test, y_predicted)

    # Plot to visualize
    draw_plot(y_test, y_predicted)

    print("Computation runs {} s".format(end-start))
