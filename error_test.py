import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras import regularizers


def test_errors(data):
    # check if data is empty
    if data.empty:
        raise ValueError("Data is empty")
    # check if date column is in correct format
    if not isinstance(data['Date'][0], pd.Timestamp):
        raise TypeError("Incorrect date format")
    # check if all rows have non-null values
    if data.isnull().values.any():
        raise ValueError("Data contains null values")


try:
    load_dotenv()
    data = pd.read_csv("bitcoin.csv")
    test_errors(data)
    data['Date'] = pd.to_datetime(data['Date'])
    data.to_csv("neural_data.csv", mode='a', header=False)
    data = data.set_index('Date').resample('T').asfreq().ffill()
    X = data[['High', 'Low', 'Open']]
    y = data['Close']
    X_train = X[:-1]
    X_test = X[-1:]
    y_train = y[:-1]
    y_test = y[-1:]
    models = [("lr", LinearRegression()),
              ("br", BayesianRidge()),
              ("svr", SVR()),
              ("rf", RandomForestRegressor())]
    for model_name, model in models:
        model.fit(X_train, y_train)
    models = [LinearRegression(), BayesianRidge(), SVR(),
              RandomForestRegressor()]
    for model in models:
        model.fit(X_train, y_train)
    lr_pred = models[0].predict(X_test)
    br_pred = models[1].predict(X_test)
    svr_pred = models[2].predict(X_test)
    rf_pred = models[3].predict(X_test)
    predictions = (lr_pred + br_pred + svr_pred + rf_pred) / 4
    print("Predicted Close Price:", predictions)
    param_grid = {'fit_intercept': [True, False],
                  'normalize': [True, False]}
    lr = LinearRegression()
    grid_search = GridSearchCV(lr, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    input_shape = X_train.shape[1]
    model = Sequential()
    model.add(Dense(64, input_dim=input_shape, activation='relu',
                    kernel_regularizer=regularizers.l1(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu',
                    kernel_regularizer=regularizers.l1(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    X_val = X_train[-1:]
    y_val = y_train[-1:]
    model.fit(X_train, y_train, validation_split=0.2,
              epochs=100, callbacks=[early_stopping])
except Exception as e:
    print("An error occurred:", e)
