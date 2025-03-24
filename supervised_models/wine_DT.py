import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error ,mean_squared_error, median_absolute_error,confusion_matrix,accuracy_score, f1_score, recall_score
from sklearn.tree import DecisionTreeClassifier

filepath = '/Users/mbencsik/Documents/python/wine_dataset/WineQT.csv'
df = pd.read_csv(filepath)
df = df.drop(columns='Id')
#print(df.columns)
X = df.drop(columns='quality')
y = df['quality']


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.to_numpy(), test_size=0.3, random_state=42)


model_DTC = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=42)
model_DTC.fit(X_train, y_train)
y_pred_DTC=model_DTC.predict(X_test)

print(" Model Evaluation Decision Tree Classifier : mean absolute error is ", mean_absolute_error(y_test,y_pred_DTC))
print(" Model Evaluation Decision Tree Classifier : mean squared  error is " , mean_squared_error(y_test,y_pred_DTC))
print(" Model Evaluation Decision Tree Classifier : median absolute error is " ,median_absolute_error(y_test,y_pred_DTC)) 
print(" Model Evaluation Decision Tree Classifier : accuracy score is ", accuracy_score(y_test, y_pred_DTC))
print(" Model Evaluation Decision Tree Classifier : f1 score (average weighted) is ", f1_score(y_test, y_pred_DTC, average = 'weighted'))
print(" Model Evaluation Decision Tree Classifier : recall score (average weighted) is ", recall_score(y_test, y_pred_DTC, average='weighted', zero_division=1))