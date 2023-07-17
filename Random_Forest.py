import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

iris = datasets.load_iris() # Downloading the iris dataset 
print(iris.target_names)
print(iris.feature_names)

df = pd.read_csv(r"C:\Users\ASUS\Desktop\Codes\Python\regression_learning\iris.csv")
df["species"].replace({'setosa' : 0, 'versicolor' : 1, 'virginica' : 2}, inplace=True) # Changing the last column in numbers

x = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] # Seperating Independent Variables
y = df['species'] # Dependent variables

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
# This will devide the DataFrame into 70% train data and 30% test data randomly.

clf = RandomForestClassifier(n_estimators=100, criterion='gini') # n_estimators : No of trees required.
clf.fit(x_train, y_train) # Training the Model
myPred = clf.predict(x_test) # Making Prediction
print(myPred)
print("Random Forest Score: ", clf.score(x_test, y_test)) # Checking the Score of the model

feature_imp = pd.Series(clf.feature_importances_, index=iris.feature_names).sort_values(ascending=False)
# Checking the importance of every Independent columns
print(feature_imp)

# Making the model again by removing 2 columns which had least feature_imp

x = df[['petal_length', 'petal_width']] 
y = df['species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

clf = RandomForestClassifier(n_estimators=100, criterion='gini') # n_estimators : No of trees required.
clf.fit(x_train, y_train)
myPred = clf.predict(x_test)
print(myPred)
print("New Random Forest Score: ", clf.score(x_test, y_test)) # Checking the score of new model. It Increases!