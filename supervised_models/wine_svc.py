import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error ,mean_squared_error, median_absolute_error,confusion_matrix,accuracy_score, f1_score, recall_score
from sklearn.svm import SVC

filepath = '/Users/mbencsik/Documents/python/wine_dataset/WineQT.csv'
df = pd.read_csv(filepath)
df = df.drop(columns='Id')
#print(df.columns)
X = df.drop(columns='quality')
y = df['quality']


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.to_numpy(), test_size=0.3, random_state=42)


model_svc = SVC(C=50, gamma='auto', kernel='rbf')
model_svc.fit(X_train, y_train)
y_pred_svc=model_svc.predict(X_test)

print(" Model Evaluation SVC : mean absolute error is ", mean_absolute_error(y_test,y_pred_svc))
print(" Model Evaluation SVC : mean squared  error is " , mean_squared_error(y_test,y_pred_svc))
print(" Model Evaluation SVC : median absolute error is " ,median_absolute_error(y_test,y_pred_svc)) 
print(" Model Evaluation SVC : accuracy score is ", accuracy_score(y_test, y_pred_svc))
print(" Model Evaluation SVC : f1 score (average weighted) is ", f1_score(y_test, y_pred_svc, average = 'weighted'))
print(" Model Evaluation SVC : recall score (average weighted) is ", recall_score(y_test, y_pred_svc, average='weighted'))