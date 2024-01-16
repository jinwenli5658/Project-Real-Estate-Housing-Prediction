# Project: Real Estate Housing Prediction (Deep Learning, ANN)
Predict the housing price using artificial neural network. Regression Type Problem 

## Data Scaling </br>
from sklearn.preprocessing import MinMaxScaler </br>
scaler = MinMaxScaler() </br>
X_scaled = scaler.fit_transform(X) </br>

## Training/Testing Data Split </br>
from sklearn.model_selection import train_test_split </br>
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.25)


## Deep Learning Model: </br>
import tensorflow.keras  </br>
from tensorflow.keras.models import Sequential </br>
from tensorflow.keras.layers import Dense </br>

model = Sequential() </br> 
model.add(Dense(100, input_dim = 7, activation = 'relu')) </br>
model.add(Dense(100, activation='relu')) </br>
model.add(Dense(100, activation='relu')) </br>
model.add(Dense(1, activation='linear')) </br>
model.summary()

model.compile(optimizer='Adam', loss = 'mean_squared_error') </br>
*Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments. </br>
epochs_hist = model.fit(X_train, y_train, epochs = 100, batch_size = 50, validation_split=0.2)  </br>
*epoch means the number of times we feed in the entire training data set and update the network weights. </br>


## Model Evaluation: </br> 
'Epoch number' vs. 'Training Loss',  </br>
'Epoch number' vs 'Validation Loss'] </br>

RMSE/MSE/MAE/R2/Adjusted R2 </br>

## Data Columns: </br>
o ida: notation for a house </br>
o date: Date house was sold  </br>
o price: Price is prediction target </br>
o bedrooms: Number of Bedrooms/House  </br>
o bathrooms: Number of bathrooms/House  </br>
o sqft_ living: square footage of the home </br>
o aft lot: square footage of the lot </br>
o floors: Total floors (levels) in house </br>
o waterfront: House which has a view to a waterfront </br>
o view: Has been viewed </br>
o grade: overall grade given to the housing unit, based on King County grading system </br>
o sqft_abovesquare: footage of house apart from basement </br>
o sqft basement: square footage of the basement </br>
o yr_ built: Built Year </br>
o yr renovated: Year when house was renovated </br>
o zipcode: zip o lat: Latitude coordinate </br>
o long: Longitude coordinate </br>
o sqft_living15: Living room area in 2015 (implies some renovations) </br>
o saft lot15: lot Size area in 2015 (implies some renovations) </br>
