import pandas as pd 
import numpy as np 
import pickle 
  
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv(url, names=column_names)
print(df.head(2))

# Renaming the target column into numbers to aid training of the model 
df['class']= df['class'].map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}) 

# splitting the data into the columns which need to be trained(X) and the target column(y) 
X = df.iloc[:, :-1] 
y = df.iloc[:, -1] 

# splitting data into training and testing data with 30 % of data as testing data respectively 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0) 

# importing the random forest classifier model and training it on the dataset 
from sklearn.ensemble import RandomForestClassifier 
classifier = RandomForestClassifier() 
classifier.fit(X_train, y_train) 

# predicting on the test dataset 
y_pred = classifier.predict(X_test) 

# finding out the accuracy 
from sklearn.metrics import accuracy_score 
score = accuracy_score(y_test, y_pred) 

# pickling the model 
pickle_out = open("classifier.pkl", "wb") 
pickle.dump(classifier, pickle_out) 
pickle_out.close()