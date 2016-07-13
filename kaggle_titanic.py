__author__ = 'Cherry_Zhang'
import numpy as np
from sklearn import tree
import pandas as pd

# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)

test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

##########################################
#Useful Functions#
##########################################

#Gets the headers(first few rows/column names) for training and test sets
trainHead = train.head()
testHead = test.head()

#Get desc(min,max,etc.) & shape(dimensions)
trainDesc = train.describe()
shape = train.shape

# absolute numbers
train["Survived"].value_counts()
# percentages
train["Survived"].value_counts(normalize = True)

# Males that survived vs males that passed away
print(train["Survived"][train["Sex"] == 'male'].value_counts())

##########################################
#Cleaning Data#
##########################################

#Convert the male and female groups to integer form
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
#Convert the same for test set
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1

#Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna("S")

#Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2

#Custom code
train["Age"] = train["Age"].fillna(train.Age.median())
test["Age"] = test["Age"].fillna(test.Age.median())

# Create the column Child and assign to 'NaN'
train["Child"] = float('NaN')
# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.
train["Child"][train["Age"] < 18] = 1
train["Child"][train["Age"] >= 18] = 0

# Create the target and features numpy arrays: target, features_one
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# Look at the importance and score of the included features
feature_importances = my_tree_one.feature_importances_
classificationAccuracy = my_tree_one.score(features_one, target)

# Impute the missing value with the median
test.Fare[152] = test.Fare.median()

# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

# Make your prediction using the test set
my_prediction = my_tree_one.predict(test_features)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)

# Write your preliminary solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_prelim.csv", index_label = ["PassengerId"])

##########################################
#Improving Predictions#
##########################################

# Create a new array with the added features: features_two
features_two = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
max_depth = 10
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth, min_samples_split, random_state = 1)
my_tree_two = my_tree_one.fit(features_two, target)

#Print the score of the new decison tree
print(my_tree_two.score(features_two, target))

# Create train_two with the newly defined feature (feature engineering family_size)
train_two = train.copy()
train_two["family_size"] = train_two["SibSp"] + train_two["Parch"] + 1

# Create a new feature set and add the new feature
features_three = train_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size"]].values

# Define the tree classifier, then fit the model
my_tree_three = tree.DecisionTreeClassifier()
my_tree_three = my_tree_three.fit(features_three, target)

# Print the score of this decision tree
print(my_tree_three.score(features_three, target))

##########################################
#Random Forest#
##########################################

# Import the `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier

# We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_forest, target)

# Print the score of the fitted random forest
print(my_forest.score(features_forest, target))

# Compute predictions on our test set features then print the length of the prediction vector
test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
pred_forest = my_forest.predict(test_features)
print(len(pred_forest))

#Request and print the `.feature_importances_` attribute
print(my_tree_two.feature_importances_)
print(my_forest.feature_importances_)

#Compute and print the mean accuracy score for both models
print(my_tree_two.score(features_two, target))
print(my_forest.score(features_two, target))

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId = np.array(test["PassengerId"]).astype(int)
solution = pd.DataFrame(pred_forest, PassengerId, columns = ["Survived"])
print(solution)

# Check that your data frame has 418 entries
print(solution.shape)

solution.to_csv("random_forest.csv", index_label = ["PassengerId"])