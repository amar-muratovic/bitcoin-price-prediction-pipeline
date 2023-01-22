import ccxt
import pandas as pd
import os
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras import regularizers
# from error_test import test_errors

load_dotenv()

data = pd.read_csv("bitcoin.csv")

# test_errors(data)

# Create a dataframe from the response
data['Date'] = pd.to_datetime(data['Date'])
data.to_csv("neural_data.csv", mode='a', header=False)

# set 'date' as the index and resample the dataframe
data = data.set_index('Date').resample('T').asfreq().ffill()

# Define the input features and target variable
X = data[['High', 'Low', 'Open']]
y = data['Close']

# Split the data into training and testing sets
X_train = X[:-1]
X_test = X[-1:]
y_train = y[:-1]
y_test = y[-1:]

# Define the models to use in the ensemble
models = [("lr", LinearRegression()),
          ("br", BayesianRidge()),
          ("svr", SVR()),
          ("rf", RandomForestRegressor())]

# Fit the models
for model_name, model in models:
    model.fit(X_train, y_train)

models = [LinearRegression(), BayesianRidge(), SVR(),
          RandomForestRegressor()]

for model in models:
    model.fit(X_train, y_train)

# Make predictions
lr_pred = models[0].predict(X_test)
br_pred = models[1].predict(X_test)
svr_pred = models[2].predict(X_test)
rf_pred = models[3].predict(X_test)

# Average the predictions
predictions = (lr_pred + br_pred + svr_pred + rf_pred) / 4

# Print the predictions
print("Predicted Close Price:", predictions)

# Define the parameter grid
param_grid = {'fit_intercept': [True, False],
              'normalize': [True, False]}

# Create a linear regression object
lr = LinearRegression()

# Create the grid search object
grid_search = GridSearchCV(lr, param_grid, cv=5)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Define the input shape
input_shape = X_train.shape[1]

# Create a sequential model
model = Sequential()

# Add the first layer to the model
model.add(Dense(64, input_dim=input_shape, activation='relu',
                kernel_regularizer=regularizers.l1(0.01)))
model.add(Dropout(0.5))

# Add the second layer to the model
model.add(Dense(32, activation='relu',
                kernel_regularizer=regularizers.l1(0.01)))
model.add(Dropout(0.5))

# Add the output layer to the model
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Split the data into training and validation sets
X_val = X_train[-1:]
y_val = y_train[-1:]

# Fit the model
model.fit(X_train, y_train, validation_split=0.2,
          epochs=100, callbacks=[early_stopping])

# Evaluate the models using mean squared error and r-squared
for model_name, model in models:
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} - MSE: {mse:.4f} - R2: {r2:.4f}")

# Fine-tune the models using cross-validation and hyperparameter tuning
for model_name, model in models:
    # Define the parameter grid
    param_grid = {'fit_intercept': [True, False], 'normalize': [True, False]}
    # Create the grid search object
    grid_search = GridSearchCV(lr, param_grid, cv=5)
    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Make predictions on the test set
y_pred = grid_search.predict(X_test)

# Compute and print the metrics
print("Mean Squared Error: {:.2f}".format(mean_squared_error(y_test, y_pred)))

# Evaluate the model on validation set
val_loss, val_acc = model.evaluate(X_val, y_val)
print("Validation Loss:", val_loss)

# Fine-tune the model using cross-validation
param_grid = {'fit_intercept': [True, False], 'normalize': [True, False]}
grid_search = GridSearchCV(lr, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Make predictions on the test set
y_pred = grid_search.predict(X_test)

# Compute and print the metrics
print("Mean Squared Error: {:.2f}".format(mean_squared_error(y_test, y_pred)))

# Evaluate the model on validation set
val_loss, val_acc = model.evaluate(X_val, y_val)
print("Validation Loss:", val_loss)

# Fine-tune the model using cross-validation and hyperparameter tuning
param_grid = {'fit_intercept': [True, False], 'normalize': [
    True, False], 'hidden_layer_sizes': [(64, 32), (128, 64)]}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Make predictions on the test set
y_pred = grid_search.predict(X_test)

# Compute and print the metrics
print("Mean Squared Error: {:.2f}".format(
    mean_squared_error(y_test, predictions)))
print("Best Parameters: ", grid_search.best_params_)
print("Best Score: {:.2f}".format(grid_search.best_score_))

# Make predictions
predictions = model.predict(X_test)

# Print the predictions
print("Predicted Close Price:", predictions)

# Print the best parameters and score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

# Define the symbol for which you want to get the historical data.
symbol = 'BTC/USD'

# Define the time period for which you want to get the historical data.
time_period = '1d'

# Get the API key
api_key = os.getenv('API_KEY')

exchange = ccxt.cryptocompare({
    'rateLimit': 2000,
    'enableRateLimit': True,
    'apiKey': api_key
})

# Get the historical data
ohlcv = exchange.fetch_ohlcv(symbol, time_period)

data = pd.DataFrame(
    ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# Add the new data to the existing dataframe
try:
    old_data = pd.read_csv('historical_data.csv')
    data = pd.concat([old_data, data])
except FileNotFoundError:
    pass

# Save the data to a csv file
data.to_csv('historical_data.csv', index=False)
