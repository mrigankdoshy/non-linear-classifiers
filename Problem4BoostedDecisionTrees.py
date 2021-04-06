from xgboost import XGBClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load data
X_test, y_test = load_svmlight_file("a9a.t")
X_train, y_train = load_svmlight_file("a9a.txt")

'''
Fit model on training data. For simplicity, let's fit based on the default values of the hyperparameters
'''

model = XGBClassifier()
model.fit(X_train, y_train)
print("\nDefault values of the hyperparameters for XGBoost are: \n", model)

# Make predictions for the test data
y_prediction = model.predict(X_test)
predictions = [round(value) for value in y_prediction]

# Evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("\nAccuracy : %.2f%%" % (accuracy * 100))

''' 
The max_depth should be changed to 5. Since the tree is shallow, the learning rate should be lowered from 0.1 to 0.05. For n_estimators < 100, the accuracy went down drastically. For values > 200, it also went down. After some hit and trial, I figured that the value of 190 gives a higher accuracy.
'''

new_model = XGBClassifier(max_depth=5, learning_rate=0.05, missing=None,
                          n_estimators=190, reg_lambda=1, objective='binary:logistic')
new_model.fit(X_train, y_train)
print("\nNew values of the hyperparameters for XGBoost are: \n", new_model)

''' 
Make predictions for the test data using model with non default parameter values 
'''

y_prediction = new_model.predict(X_test)
predictions = [round(value) for value in y_prediction]

# Evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("\nAccuracy : %.2f%%" % (accuracy * 100))
