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
print("Default values of the hyperparameters for XGBoost are: \n", model)

# Make predictions for the test data
y_prediction = model.predict(X_test)
predictions = [round(value) for value in y_prediction]

# Evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy : %.2f%%" % (accuracy * 100))
