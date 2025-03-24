# Comparison of the 🍷 Wine dataset analysis with supervised machine learning models 🚀

The original dataset: 

kaggle datasets download yasserh/wine-quality-dataset


This version of the wine dataset contains 10 feature columns and 1 target column. During preprocessing the features were scaled with 
### MinMaxScaler

Feature columns:
  - fixed acidity
  - volatile acidity
  - citric acid
  - residual sugar
  - chlorides
  - free sulfur dioxide
  - total sulfur dioxide
  - density
  - pH
  - sulphates
  - alcohol

  
Target column: 
  - quality


# Supervised Models
The following Supervised Machine Learning were applied

### 🕐 Linear Regression

### 🕑 Logistic Regression

### 🕒 Decision Tree Classifier

### 🕓 K Neighbour Classifier

### 🕔 Support Vector Machine SVM

### 🕕 AdaBoost


# Evaluation metrics
📝 The algorithms were evaluated for accuracy, precision, recall, f1-score, mean standard error and mean absolute error.

# Visualization
📈 For comparison, the results were tabulated and visualized with seaborn package


# Results
📓 The metrics are somewhat low, which could be partially explained by the high number of classes in the quality target. Next steps could be to implement dimensionality reduction with clustering algorithms to assess if the metrics improve. 

