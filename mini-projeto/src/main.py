import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from preprocess import preprocess_data
from classifiers import train_perceptron, train_linear_regression, train_logistic_regression

def filter_digits(df, digit1, digit2):
    return df[(df['label'] == digit1) | (df['label'] == digit2)]

def plot_data(X, y, title):
    plt.figure()
    for label, color in zip([1, 5], ['blue', 'red']):
        subset = X[y == label]
        plt.scatter(subset[:, 0], subset[:, 1], color=color, label=f"Dígito {label}")
    plt.title(title)
    plt.xlabel('Intensidade')
    plt.ylabel('Simetria')
    plt.legend()
    plt.show()

train_df = pd.read_csv('train_redu.csv')
test_df = pd.read_csv('test_redu.csv')

# Filtrar para os dígitos 1 e 5
train_1x5 = filter_digits(train_df, 1, 5)
test_1x5 = filter_digits(test_df, 1, 5)

X_train = train_1x5[['intensidade', 'simetria']].values
y_train = train_1x5['label'].values

X_test = test_1x5[['intensidade', 'simetria']].values
y_test = test_1x5['label'].values

plot_data(X_train, y_train, "Dígitos 1 e 5 (Treino)")

# Treinar classificadores
clf_perceptron = train_perceptron(X_train, y_train)
clf_linear = train_linear_regression(X_train, y_train)
clf_logistic = train_logistic_regression(X_train, y_train)

# Avaliar classificadores
for clf, name in zip([clf_perceptron, clf_linear, clf_logistic],
                     ['Perceptron', 'Regressão Linear', 'Regressão Logística']):
    y_pred = clf.predict(X_test)
    print(f"Relatório de Classificação - {name}")
    print(confusion_matrix(y_test, y_pred.round()))
    print(classification_report(y_test, y_pred.round(), zero_division=0))
