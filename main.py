import pandas as pd
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras import regularizers

data = pd.read_csv("bitcoin.csv")

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

# Make predictions
predictions = model.predict(X_test)

# Print the predictions
print("Predicted Close Price:", predictions)

# Print the best parameters and score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)
