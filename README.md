# Project-Real-Estate-Housing-Prediction
Predict the housing price using artificial neural network.

Regression Type Problem </br>
Data Scaling </br>
**from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)**

Data Columns: </br>
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
