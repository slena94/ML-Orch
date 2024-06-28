import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

input_file = "paris_weather_data_4_years.csv"
model_file = "temperature_prediction_model.pkl"

data = pd.read_csv(input_file)

data['date'] = pd.to_datetime(data['date'])

data['temperature_2m'].fillna(method='ffill', inplace=True)

data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['hour'] = data['date'].dt.hour

# Décaler la température de 24 heures pour créer la variable cible
data['temperature_next_day'] = data['temperature_2m'].shift(-24)

# Supprimer les dernières 24 lignes car elles n'ont pas de valeur de température pour le jour suivant
data.dropna(subset=['temperature_next_day'], inplace=True)

X = data[['year', 'month', 'day', 'hour', 'temperature_2m']]
y = data['temperature_next_day']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Enregistrer le modèle dans un fichier
joblib.dump(model, model_file)
print(f"Model has been saved to {model_file}")