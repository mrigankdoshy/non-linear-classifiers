from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Load data
X_test, y_test = load_svmlight_file("a9a.t")
X_train, y_train = load_svmlight_file("a9a.txt")

'''
Fit model on training data. For simplicity, let's fit based on the default values of the hyperparameters
'''

model = RandomForestClassifier()
model.fit(X_train, y_train)
print("\nDefault values of the hyperparameters for Random Forest Classifier are: \n", model.get_params())

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

n_estimators = [50, 100, 200, 300]
bootstrap = [True, False]
max_depth = [None, 500, 1000]
min_impurity_decrease = [0.0, .05, 0.1, 0.2]
min_samples_leaf = [1, 2, 10, 50, 100]

hyperparameters = []
for n_estimate, boot, depth, min_impurity, min_samples in product(n_estimators, bootstrap, max_depth, min_impurity_decrease, min_samples_leaf):
    hyperparameters.append(
        [n_estimate, boot, depth, min_impurity, min_samples])

best_accuracy = 0
for parameter in hyperparameters:
    parameters = {'n_estimators': parameter[0], 'bootstrap': parameter[1],
                  'max_depth': parameter[2], 'min_impurity_decrease': parameter[3], 'min_samples_leaf': parameter[4]}
    model = RandomForestClassifier(n_estimators=parameter[0],
                                   bootstrap=parameter[1],
                                   max_depth=parameter[2], min_impurity_decrease=parameter[3], min_samples_leaf=parameter[4])

    kfold = KFold()
    cross_val_scores = cross_val_score(model, X_train, y_train, cv=kfold)
    accuracy = cross_val_scores.mean() * 100

    if (best_accuracy < accuracy):
        best_accuracy = accuracy
        best_model = parameters

print("\nBest accuracy that we can get: ", accuracy)
print("\nThe model that gives this is: ", best_model)

'''
The max_depth should be changed to 5. The learning rate should be 0.2. For n_estimators < 100, the accuracy went down drastically. For values > 250, it also went down. After some hit and trial, I figured that the value of 190 or 200 gives a higher accuracy.
'''

# new_model = RandomForestClassifier(n_estimators=parameter[0],
#                                    bootstrap=parameter[1],
#                                    max_depth=parameter[2], min_impurity_decrease=parameter[3], min_samples_leaf=parameter[4])
# new_model.fit(X_train, y_train)
# print("\nNew values of the hyperparameters for XGBoost are: \n", new_model)

# kfold = KFold()
# cross_val_scores = cross_val_score(new_model, X_train, new_y_train, cv=kfold)
# accuracy = cross_val_scores.mean() * 100

# print("\nAccuracy of the new model: ", accuracy)

# print("\nCross Validation Training Error Rate for the new model: ",
#       1-cross_val_scores.mean())

# print("\nTest Error Rate for the new model: ",
#       1-new_model.score(X_test, y_test))
