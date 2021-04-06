from xgboost import XGBClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from itertools import product
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
print("\nAccuracy: %.2f%%" % (accuracy * 100))

''' 
Tuning hyperparameters.
Make lists of values and get all possible combinations
If I add more entries, it takes way too long to compute. 

WARNING: This takes 1 hour compute on a MacBook Pro: 2.8 GHz Quad-Core Intel Core i7 with 16GB Memory. 
Please comment from line 46 to 75 to run final answer directly.
'''

new_y_train = []
for i in y_train:
    if (i == -1):
        new_y_train.append(0)
    else:
        new_y_train.append(i)

max_depth = [3, 4, 5]
learning_rate = [0.05, 0.1, 0.2]
missing = [None, 0]
n_estimators = [100, 200, 300]
reg_lambda = [0.0, 1.0]
objective = ('binary:logistic', 'binary:logitraw', 'binary:hinge')

hyperparameters = []
for depth, rate, miss, n_estimate, lam, obj in product(max_depth, learning_rate, missing, n_estimators, reg_lambda, objective):
    hyperparameters.append(
        [depth, rate, miss, n_estimate, lam, obj])

best_accuracy = 0
for parameter in hyperparameters:
    parameters = {'max_depth': parameter[0], 'learning_rate': parameter[1],
                  'missing': parameter[2], 'n_estimators': parameter[3], 'reg_lambda': parameter[4], 'objective': parameter[5]}
    model = XGBClassifier(max_depth=parameter[0], learning_rate=parameter[1],
                          missing=parameter[2], n_estimators=parameter[3], reg_lambda=parameter[4], objective=parameter[5])

    kfold = KFold()
    cross_val_scores = cross_val_score(model, X_train, new_y_train, cv=kfold)
    accuracy = cross_val_scores.mean() * 100

    if (best_accuracy < accuracy):
        best_accuracy = accuracy
        best_model = parameters

print("\nBest accuracy that we can get: ", accuracy)
print("\nThe model that gives this is: ", best_model)

''' 
The max_depth should be changed to 5. Since the tree is shallow, the learning rate should be lowered from 0.1 to 0.05. For n_estimators < 100, the accuracy went down drastically. For values > 200, it also went down. After some hit and trial, I figured that the value of 190 gives a higher accuracy.
'''

new_model = XGBClassifier(max_depth=3, learning_rate=0.2, missing=None,
                          n_estimators=200, reg_lambda=1, objective='binary:logistic')
new_model.fit(X_train, y_train)
print("\nNew values of the hyperparameters for XGBoost are: \n", new_model)

kfold = KFold()
cross_val_scores = cross_val_score(new_model, X_train, new_y_train, cv=kfold)
accuracy = cross_val_scores.mean() * 100

print("\nAccuracy of the new model: ", accuracy)

print("\nCross Validation Training Error Rate for the new model: {0}".format(
    1-cross_val_scores.mean()))

print("\nTest Error Rate for the new model: {0}".format(
    1-new_model.score(X_test, y_test)))
