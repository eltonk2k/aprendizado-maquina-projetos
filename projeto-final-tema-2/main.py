import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from sklearn.metrics import confusion_matrix, classification_report


# Carregar os dados
data = pd.read_csv("train.csv")


data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
data['Fare'] = data['Fare'].fillna(data['Fare'].median())


# Selecionar features e alvo
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'

X = data[features]
y = data[target]

# Transformar dados categóricos
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['Age', 'Fare', 'SibSp', 'Parch']),
    ('cat', OneHotEncoder(), ['Pclass', 'Sex', 'Embarked'])
])

X = preprocessor.fit_transform(X)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar o modelo MLP
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinar o modelo
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# Avaliar o modelo
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Matriz de confusao
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:")
print(conf_matrix)

# Relatório de classificacao
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

# Curvas de erro
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Treinamento')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Erro Médio por Época')
plt.xlabel('Épocas')
plt.ylabel('Erro')
plt.legend()
plt.show()
