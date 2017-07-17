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

from keras.layers.core import Reshape
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, LSTM, Dropout, Activation, TimeDistributed, Lambda, RepeatVector, Flatten, Reshape, Input
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils


def load_data():
    # Load relevant datas ( In this case 4 data to form 4 features)
    data_0 = glob.glob('INSERT DIR')

    itemIndex = [0,1,4,5]

    all_df = []

    for i in itemIndex:
        df0 = pd.read_csv(data_0[i], header=None)
        if df0.isnull().sum().sum() > 0:
            df0 = df0.dropna(axis = 0, how= 'any')
        df0 = df0[2].tolist()

        all_df.append(df0)

    return all_df

def create_train_test(sc, input_data, look_back, jump_steps, jump_range):

    jump_limit = jump_steps + jump_range - 2 + 3
    jump_steps -= 1

    # Split datasets into 4/5 training and 1/5 test sets
    input_data_length = len(input_data)
    train_test_boundary = int(input_data_length * 4 / 5)

    # Create train datasets and scale it
    train_data = input_data[:train_test_boundary]
    train_data = [[elem] for elem in train_data]
    train_data_scaled = sc.fit_transform(train_data)

    train_data_length = len(train_data)
    x_train = []
    y_train = []
    for i in range(look_back, train_data_length - jump_limit):
        x_train.append(train_data_scaled[i - look_back : i, 0])
        y_train.append(train_data_scaled[ (i + jump_steps): i + (jump_steps + jump_range), 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    temp_input_data = [[elem] for elem in input_data]
    input_data_scaled = sc.fit_transform(temp_input_data)

    x_test = []
    y_test = []
    for j in range(train_test_boundary + 14, input_data_length - jump_limit):
        x_test.append(input_data_scaled[j - look_back : j, 0])
        y_test.append(input_data_scaled[ (j + jump_steps): j + (jump_steps + jump_range), 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    return (x_train, y_train), (x_test, y_test)

def three_dim_one_hot_encode(sequence):
    encoding = []
    for value in sequence:
        # Create 16 x 9 matrices
        sub_encoding = [0 for _ in range(17)]

        # Check if value is 1
        tokens = list(str(value))
        n = len(tokens[2:]) + 1
        sub_encoding[0] = int(tokens[0])
        sub_encoding[1:n] = [int(elem) for elem in tokens[2:]]
        encoding.append(sub_encoding)

    return encoding

def build_model(look_back, xshape, yshape, batch_size):
    # Building models

    encoding_dim = 30
    inputs = Input(shape=( xshape[1], xshape[2] ) )
    encoded = LSTM(units = 300)(inputs)
    encoded = RepeatVector(x_train.shape[1])(encoded)
    encoded = LSTM(units = 150, return_sequences = True)(encoded)
    encoded = LSTM(units = 70, return_sequences = True)(encoded)
    encoded = LSTM(units = encoding_dim, return_sequences = True)(encoded)

    decoded = LSTM(units = 70, return_sequences = True)(encoded)
    decoded = LSTM(units = 150, return_sequences = True)(decoded)
    decoded = LSTM(units = 300, return_sequences = True)(decoded)
    decoded = LSTM(units = xshape[2], return_sequences = True)(encoded)

    autoencoder = Model(inputs, decoded)
    print(autoencoder.summary())

    encoder = Model(inputs, encoded)
    encoded_input = Input(shape=(None,encoding_dim))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    adam = Adam(lr = 0.0005)
    autoencoder.compile(optimizer = adam, loss = 'mean_squared_error')

    return autoencoder, encoder, decoder

def load_model():
    # Building models
    with open('YOUR SAVED MODEL JSON', 'r') as json_file:
        loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('YOUR SAVED WEIGHTS')
    adam = Adam(lr = 0.0005)
    loaded_model.compile(optimizer = adam, loss = 'mean_squared_error')
    print(loaded_model.summary())

    return loaded_model

def draw_plot(y_real, y_predicted, start_point, end_point, jump):
    # Visualising the results
    for a in range(start_point, end_point, jump):
        plt.plot(y_real[a, :], color = 'red', label = 'Real Data Projection')
        plt.plot(y_predicted[a, :], color = 'blue', label = 'Predicted Data Projection')\
        plt.xlabel('Time')
        plt.ylabel('Bytes')
        rmse = math.sqrt(mean_squared_error(y_real[a, :], y_predicted[a, :]))
        for k in range(len(y_real[a, :])):
            if( abs(y_real[a, k] - y_predicted[a, k]) > rmse):
                plt.plot(k, y_real[a, k], "o", markersize=5, markeredgewidth=1, markeredgecolor='g', markerfacecolor='None')
        plt.show()

def save_weights_layout(model):
    model_json = model.to_json()
    with open("Saved Model Name JSON", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("Saved Model Weights H5")


if __name__ == "__main__":
    # Declare variables
    lag_steps = 120   # Using [(t-120), (t-119) ... (t-1)]
    bsize = 64      # standard batch size for every weight update
    iterations = 3 # Number of epochs, usually in the range of 50 - 1000, making it 3 for testing purpose on CPU
    forecast_steps = 10
    forecase_range = 5

    # Regularization for overfit circumstances
    #reg = [L1L2(l1=0.0, l2=0.0), L1L2(l1=0.01, l2=0.0), L1L2(l1=0.0, l2=0.01), L1L2(l1=0.01, l2=0.01)]

    # Scaler for scaling data from arbitrary range into 0 - 1
    sc = MinMaxScaler(feature_range = (0, 1))

    # Start Measuring computation time
    start = time.time()

    # Loading input data from folders
    input_data = load_data()

    train_sets, test_sets =  create_train_test(sc, input_data[0], lag_steps, forecast_steps, forecase_range)
    matx_train, maty_train = train_sets[0], train_sets[1]
    matx_test, maty_test = test_sets[0], test_sets[1]

    matx_train = np.reshape(matx_train, (matx_train.shape[0], matx_train.shape[1], 1))
    maty_train = np.reshape(maty_train, (maty_train.shape[0], maty_train.shape[1], 1))
    matx_test = np.reshape(matx_test, (matx_test.shape[0], matx_test.shape[1], 1))
    maty_test = np.reshape(maty_test, (maty_test.shape[0], maty_test.shape[1], 1))

    for index, elem in enumerate(input_data[1:]):
        # Split data into train and test sets
        train_sets, test_sets =  create_train_test(sc, elem, lag_steps, forecast_steps, forecase_range)
        x_train, y_train = train_sets[0], train_sets[1]
        x_test, y_test = test_sets[0], test_sets[1]

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))

        matx_train = np.dstack((matx_train, x_train))
        maty_train = np.dstack((maty_train, y_train))
        matx_test = np.dstack((matx_test, x_test))
        maty_test = np.dstack((maty_test, y_test))

    # Build LSTM model
    autoencoder, encoder, decoder = build_model(lag_steps, matx_train.shape, maty_train.shape, bsize)

    # Training train_datasets
    autoencoder.fit(matx_train, matx_train, epochs = iterations, batch_size = bsize)

    encoded_seq = encoder.predict(matx_train)
    decoded_seq = decoder.predict(encoded_seq)

    # Just checking on the first feature (X, Y, 0) -> (number of batches, timesteps, number of features)
    y_real = sc.inverse_transform(matx_train[:,:,0])

    decoded_seq1 = decoded_seq[:,:,0]
    y_predicted = sc.inverse_transform(decoded_seq1)
    #save_weights_layout(model)
    # Generate predicted resutlts
    # y_predicted = model.predict(x_test)
    print("==========1")
    print(decoded_seq1.shape)
    print(y_real)
    print(y_predicted)
    print(y_predicted.shape)

    draw_plot(y_real, y_predicted, 5500, 6500, 50)

    end = time.time()

    print("Computation runs {} s".format(end-start))
