import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_models():
    filepath = '/Users/mbencsik/Documents/python/wine_dataset/WineQT.csv'
    df = pd.read_csv(filepath)
    df = df.drop(columns='Id')
    #print(df.columns)
    X = df.drop(columns='quality')
    y = df['quality']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.to_numpy(), test_size=0.3, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=42),
        'K Neighbors': KNeighborsClassifier(n_neighbors=3),
        'SVC': SVC(C=50, gamma='auto', kernel='rbf'),
        'AdaBoost': AdaBoostClassifier(n_estimators = 10, learning_rate = 0.001, random_state = 42)
    }
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        if name == 'Linear Regression':
            y_pred = model.predict(X_test)
            y_pred = np.round(y_pred).astype(int)
            y_pred = np.clip(y_pred, y.min(), y.max())
        else:
            y_pred = model.predict(X_test)

        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision (weighted)': precision_score(y_test, y_pred, average='weighted', zero_division=1),
            'Recall (weighted)': recall_score(y_test, y_pred, average='weighted', zero_division=1),
            'F1 (weighted)': f1_score(y_test, y_pred, average='weighted'),
            'Mean absolute error': mean_absolute_error(y_test,y_pred),
            'Mean squared  error': mean_squared_error(y_test,y_pred)
            })

    df_results = pd.DataFrame(results)
    return df_results

models_eval = evaluate_models()
#print(models_eval)
df_melted = models_eval.melt(id_vars='Model', var_name='Metric', value_name='Score')
plt.figure(figsize=(10, 6))
ax = sns.lineplot(data=df_melted, x='Model', y='Score', hue='Metric')
plt.title("Supervised Model Comparison of the Wine Dataset")
ax.set(xlabel=None)
plt.show()


