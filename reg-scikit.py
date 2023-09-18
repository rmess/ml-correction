# Importation des bibliothèques nécessaires
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler


data = pd.read_csv('insurance.csv')

# 2. Prétraitement des données

# Conversion des variables catégorielles en variables numériques avec le codage à chaud (one-hot encoding)
encoder = OneHotEncoder(drop='first')  # drop='first' pour éviter la multi-collinéarité
categorical_features = ['sex', 'smoker', 'region']
encoded_features = encoder.fit_transform(data[categorical_features]).toarray()
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))

# Fusion des données encodées avec le dataset original
data = pd.concat([data, encoded_df], axis=1)
data = data.drop(columns=categorical_features)  # Suppression des colonnes catégorielles originales

# Séparation des caractéristiques (features) et de la variable cible (target)
X = data.drop(columns='charges')
y = data['charges']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation des données pour améliorer la performance de la régression linéaire
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Entraînement du modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Prédiction et évaluation du modèle
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Erreur quadratique moyenne (MSE) : {mse}")

# 5. Visualisation de la régression linéaire
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)  # Points réels vs prédits
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Ligne de référence parfaite
plt.title('Valeurs réelles vs Valeurs prédites')
plt.xlabel('Valeurs réelles')
plt.ylabel('Valeurs prédites')
plt.grid(True)
plt.show()

# Si vous souhaitez obtenir les coefficients du modèle :
coefficients = model.coef_
print(f"Coefficients : {coefficients}")
