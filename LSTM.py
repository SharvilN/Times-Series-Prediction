# from numpy.random import seed
# seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from keras.models import load_model
from matplotlib import pyplot
import numpy
import sys
import os.path


period = 25

class Predictions(Callback):
    def __init__(self, pr, ts, p):
        super(Callback, self).__init__()
        self.period = pr
        self.ts = ts
        self.a = p

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:
            prediction = list()
            for k in range(len(self.ts)):
                # make one-step forecast
                X, y = self.ts[k, 0:-1], self.ts[k, -1]
                yhat = forecast_lstm(self.model, 1, X)
                # invert scaling
                yhat = invert_scale(scaler, X, yhat)
                # invert differencing
                yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - k)
                # store forecast
                prediction.append(yhat)
                expected = raw_values[len(train) + k + 1]
                print('Month=%d, Predicted=%f, Expected=%f' % (k + 1, yhat, expected))
                # file.write('Month=%d, Predicted=%f, Expected=%f' % (i + 1, yhat, expected))

            # report performance
            # rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
            # print('Test RMSE: %.3f' % rmse)
            # line plot of observed vs predicted
            print('Test MSE: %.3f' % numpy.mean((raw_values[self.a + 1:] - prediction) ** 2))
            rms = sqrt(mean_squared_error(raw_values[self.a + 1:], prediction))
            print('Test RMSE: %.3f' % rms)
            f1 = open("res_call_512.txt", "a")
            f1.write("epochs = " + str(epoch + 1) + "\trmse = " + str(rms) + "\n")
            f1.close()
            f1 = open("checkpoint.txt", "w")
            if epoch < 10:
                fname = "weights-improvement-512-0" + str(epoch) + ".hdf5"
            else:
                fname = "weights-improvement-512-" + str(epoch) + ".hdf5"
            f1.write(fname + "\n" + str(epoch))
            f1.close()


# date-time parsing function for loading the dataset
# def parser(x):
#     return datetime.strptime('190' + x, '%Y-%m')


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]


# fit an LSTM network to training data
def fit_lstm(train, test_scaled, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    print("X = " , X)
    print("y = " , y)

    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    filepath = "weights-improvement-512-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, verbose=0, save_best_only=False, save_weights_only=False, mode='auto'
                                 , period=period)
    checkpoint2 = Predictions(period, test_scaled, a)
    callbacks_list = list()
    callbacks_list.append(checkpoint)
    callbacks_list.append(checkpoint2)
    model.fit(X, y, nb_epoch=nb_epoch, batch_size=batch_size, callbacks=callbacks_list, verbose=0, shuffle=False)
    # for i in range(int(nb_epoch / 10)):
    #     model.fit(X, y, nb_epoch=10, batch_size=batch_size, callbacks=callbacks_list, verbose=0, shuffle=False)
    #     train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    #     model.predict(train_reshaped, batch_size=1)
    #     predictions = list()
    #     for j in range(len(test_scaled)):
    #         # make one-step forecast
    #         X, y = test_scaled[j, 0:-1], test_scaled[j, -1]
    #         yhat = forecast_lstm(model, 1, X)
    #         # invert scaling
    #         yhat = invert_scale(scaler, X, yhat)
    #         # invert differencing
    #         yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - j)
    #         # store forecast
    #         predictions.append(yhat)
    #         expected = raw_values[len(train) + j + 1]
    #         print('Month=%d, Predicted=%f, Expected=%f' % (j + 1, yhat, expected))
    #         # file.write('Month=%d, Predicted=%f, Expected=%f' % (i + 1, yhat, expected))
    #
    #     # report performance
    #     # rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
    #     # print('Test RMSE: %.3f' % rmse)
    #     # line plot of observed vs predicted
    #     print('Test MSE: %.3f' % numpy.mean((raw_values[a + 1:] - predictions) ** 2))
    #     rmse = sqrt(mean_squared_error(raw_values[a + 1:], predictions))
    #     print('Test RMSE: %.3f' % rmse)
    #     pyplot.plot(raw_values[a + 1:])
    #     pyplot.plot(predictions)
    #     # pyplot.show()
    #
    #     # model.reset_states()
    #     print('epoch = ' + str(i))
    return model





# load dataset
# series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
dateparse = lambda dates: datetime.strptime(dates, '%Y-%m-%d')
series = read_csv('BRK-A.csv', parse_dates=["Date"], index_col="Date", date_parser=dateparse)
# print data.head()
# fix random seed for reproducibility
# numpy.random.seed(7)

# transform data to be stationary
raw_values = series.values
diff_values = difference(raw_values, 1)

# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# split data into train and test-sets
a = int(len(raw_values) * 0.66 )
train, test = supervised_values[0:a], supervised_values[a:]

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

# file = open('result.txt', 'a')

# fit the model
# for k in range(100000000):
epochs = 2500

index = period - 1
try:
    index = int(sys.argv[1])
    if index < 10:
        fname = "weights-improvement-512-0" + str(index - 1) + ".hdf5"
    else:
        fname = "weights-improvement-512-" + str(index - 1) + ".hdf5"
    model = load_model(fname)
    model.compile(loss='mean_squared_error', optimizer='adam')
    filepath = "weights-improvement-512-{epoch:02d}" + "" + ".hdf5"
    checkpoint = ModelCheckpoint(filepath, verbose=0, save_best_only=False, save_weights_only=False, mode='auto'
                                 , period=period)
    checkpoint2 = Predictions(period, test_scaled, a)
    callbacks_list = list()
    callbacks_list.append(checkpoint)
    callbacks_list.append(checkpoint2)
    X, y = train_scaled[:, 0:-1], train_scaled[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model.fit(X, y, nb_epoch=epochs, batch_size=1, callbacks=callbacks_list, verbose=0,
              shuffle=False, initial_epoch=index)
except (IndexError, ValueError):
    if index < 10:
        fname = "weights-improvement-512-0" + str(index) + ".hdf5"
    else:
        fname = "weights-improvement-512-" + str(index) + ".hdf5"
    if os.path.isfile(fname):
        # while os.path.isfile(fname):
        #     index = index + period
        #     if index < 10:
        #         fname = "weights-improvement-0" + str(index) + ".hdf5"
        #     else:
        #         fname = "weights-improvement-" + str(index) + ".hdf5"
        # index = index - period
        # if index < 10:
        #     fname = "weights-improvement-0" + str(index) + ".hdf5"
        # else:
        #     fname = "weights-improvement-" + str(index) + ".hdf5"
        f1 = open("checkpoint.txt", "r")
        fname = f1.readline()
        fname = fname[:-1]
        # print(fname)
        index = f1.read()
        index = int(index)
        index = index + 1
        model = load_model(fname)
        model.compile(loss='mean_squared_error', optimizer='adam')
        filepath = "weights-improvement-512-{epoch:02d}" + "" + ".hdf5"
        checkpoint = ModelCheckpoint(filepath, verbose=0, save_best_only=False, save_weights_only=False, mode='auto'
                                     , period=period)
        checkpoint2 = Predictions(period, test_scaled, a)
        callbacks_list = list()
        callbacks_list.append(checkpoint)
        callbacks_list.append(checkpoint2)
        X, y = train_scaled[:, 0:-1], train_scaled[:, -1]
        X = X.reshape(X.shape[0], 1, X.shape[1])
        model.fit(X, y, nb_epoch=epochs, batch_size=1, callbacks=callbacks_list, verbose=0,
                  shuffle=False, initial_epoch=index)
    else:
        lstm_model = fit_lstm(train_scaled, test_scaled, 1, epochs, 512)
# forecast the entire training dataset to build up state for forecasting

# for j in range(int(epochs / period)):
#
#     index = ((j + 1) * period) - 1
#     if(index < 10):
#         lstm_model.load_weights("weights-improvement-0" + str(index) + ".hdf5")
#     else:
#         lstm_model.load_weights("weights-improvement-" + str(index) + ".hdf5")
#     lstm_model.compile(loss='mean_squared_error', optimizer='adam')
#
#     predictions = list()
#     for i in range(len(test_scaled)):
#         # make one-step forecast
#         X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
#         yhat = forecast_lstm(lstm_model, 1, X)
#         # invert scaling
#         yhat = invert_scale(scaler, X, yhat)
#         # invert differencing
#         yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
#         # store forecast
#         predictions.append(yhat)
#         expected = raw_values[len(train) + i + 1]
#         print('Month=%d, Predicted=%f, Expected=%f' % (i + 1, yhat, expected))
#         # file.write('Month=%d, Predicted=%f, Expected=%f' % (i + 1, yhat, expected))
#
#     # report performance
#     # rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
#     # print('Test RMSE: %.3f' % rmse)
#     # line plot of observed vs predicted
#     print('Test MSE: %.3f' % numpy.mean((raw_values[a + 1:] - predictions) ** 2))
#     rmse = sqrt(mean_squared_error(raw_values[a + 1:], predictions))
#     print('Test RMSE: %.3f' % rmse)
#     f = open("res.txt", "a")
#     f.write("epochs = " + str((j + 1) * period) + "\trmse = " + str(rmse) + "\n")
#     f.close()
#     pyplot.plot(raw_values[a + 1:])
#     pyplot.plot(predictions)
