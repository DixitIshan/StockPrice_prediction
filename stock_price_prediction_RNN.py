import os
import math

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation, Bidirectional, LSTM, GRU

def to_1dimension(df, step_size):
    X, y = [], []
    for i in range(len(df)-step_size-1):
        data = df[i:(i+step_size), 0]
        X.append(data)
        y.append(df[i + step_size, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    return X, y
     
    
def plot_series(values, xlabel=None, ylabel=None, color='b', legend=None):
    xx = np.arange(1, len(values) + 1, 1)
    plt.plot(xx, values, color, label=legend)
    plt.legend(loc = 'upper left')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

    
def plot_series_prediction(true_values, train_predict, test_predict, time_ahead=1, title=None,
                           xlabel=None, ylabel=None, color=['green','red','blue'], legend=[None,None,None]):    
    TOOLS = 'pan,wheel_zoom,box_zoom,reset,save,box_select'
    #x axis
    xx = np.array(range(true_values.shape[0]))
    xx1 = np.array(range(time_ahead,len(train_predict)+time_ahead))
    xx2 = np.array(range(len(train_predict)+(time_ahead*2)+1,len(true_values)-1))
    
    #figure
    p = figure(title=title, tools=TOOLS)
    p.line(xx, true_values.squeeze(), legend=legend[0], line_color=color[0], line_width=2)
    p.line(xx1, train_predict.squeeze(), legend=legend[1], line_color=color[1], line_width=1)    
    p.line(xx2, test_predict.squeeze(), legend=legend[2], line_color=color[2], line_width=1)
    p.axis[0].axis_label = xlabel
    p.axis[1].axis_label = ylabel
    p.legend.location = "top_left"
    show(p)

DATA = os.path.join('data', 'MSFT_2012_2017.csv')
EPOCHS = 50
TEST_SIZE = 0.3
TIME_AHEAD = 1 #prediction step
BATCH_SIZE = 1
UNITS = 25

df = pd.read_csv(DATA)
df = df.drop(['Adj Close', 'Volume'], axis=1)
print(df.shape)
print(df.head())

mean_price = df.mean(axis = 1)
plot_series(mean_price, xlabel='Days', ylabel='Mean value of Microsoft Stock', color='b', legend='Mean price')


scaler = MinMaxScaler(feature_range=(0, 1)) #other typical scale values are -1,1
mean_price = scaler.fit_transform(np.reshape(mean_price.values, (len(mean_price),1)))

train, test = train_test_split(mean_price, test_size=TEST_SIZE, shuffle=False)
print(train.shape)
print(test.shape)

X_train, y_train = to_1dimension(train, TIME_AHEAD)
X_test, y_test = to_1dimension(test, TIME_AHEAD)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

def create_symbol(model_name='LSTM', units=10, activation='linear', time_ahead=1):
    model = Sequential()
    if model_name == 'LSTM':
        model.add(LSTM(units, input_shape=(1, time_ahead)))
    elif model_name == 'BiLSTM':
        model.add(Bidirectional(LSTM(units), input_shape=(1, time_ahead)))
    elif model_name == 'GRU':
        model.add(GRU(units, input_shape=(1, time_ahead)))
    else:
        raise ValueError("Wrong model name")
    model.add(Dense(1))
    model.add(Activation('linear'))
    return model

################# LSTM #################
model = create_symbol(model_name='LSTM', units=UNITS, time_ahead=TIME_AHEAD)
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)

y_test_inv = scaler.inverse_transform([y_test])
mean_price_inv = scaler.inverse_transform(mean_price)

pred_test = model.predict(X_test) #pred_test.shape = (num_rows, TIME_AHEAD)
pred_test = scaler.inverse_transform(pred_test)
score = math.sqrt(mean_squared_error(y_test_inv[0], pred_test[:,0]))
print('Test RMSE: %.2f' % (score))

pred_train = model.predict(X_train)
pred_train = scaler.inverse_transform(pred_train)
plot_series_prediction(mean_price_inv, pred_train, pred_test, time_ahead=TIME_AHEAD,
	title='LSTM prediction', xlabel='Days', ylabel='Value of Microsoft Stock', 
	legend=['True value','Training set','Test prediction'])


################# BiLSTM #################
model = create_symbol(model_name='BiLSTM', units=UNITS, time_ahead=TIME_AHEAD)
model.compile(optimizer='adam', loss='mean_squared_error')


model.fit(X_train, y_train, epochs=EPOCHS, batch_size=1, verbose=2)

pred_test = model.predict(X_test)
pred_test = scaler.inverse_transform(pred_test)
score = math.sqrt(mean_squared_error(y_test_inv[0], pred_test[:,0]))
print('Test RMSE: %.2f' % (score))

pred_train = model.predict(X_train)
pred_train = scaler.inverse_transform(pred_train)
plot_series_prediction(mean_price_inv, pred_train, pred_test, time_ahead=TIME_AHEAD,
	title='BiLSTM prediction', xlabel='Days', ylabel='Value of Microsoft Stock', 
	legend=['True value','Training set','Test prediction'])

################# GRU #################
model = create_symbol(model_name='GRU', units=UNITS, time_ahead=TIME_AHEAD)
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=EPOCHS, batch_size=1, verbose=2)

pred_test = model.predict(X_test)
pred_test = scaler.inverse_transform(pred_test)
score = math.sqrt(mean_squared_error(y_test_inv[0], pred_test[:,0]))
print('Test RMSE: %.2f' % (score))

pred_train = model.predict(X_train)
pred_train = scaler.inverse_transform(pred_train)
plot_series_prediction(mean_price_inv, pred_train, pred_test, time_ahead=TIME_AHEAD,
	title='GRU prediction', xlabel='Days', ylabel='Value of Microsoft Stock', 
	legend=['True value','Training set','Test prediction'])
