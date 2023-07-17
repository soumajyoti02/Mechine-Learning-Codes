import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


df = pd.read_csv(r"C:\Users\ASUS\Desktop\Codes\Python\regression_learning\iris.csv")

x = df.iloc[:, 0:4] # Taking 0th to 4th Column i.e. Independent Column
y = df.iloc[:, 4] # Taking 5th column : Dependent Column

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = SVC(kernel='linear') # Creating the Vector mechine instance
model.fit(x_train, y_train) # Training the model

pred = model.predict(x_test) # Testing my model
print(x_test) # This is the randomly Generated test data by train_test_split
print(pred)
print("Score of SVM Model: ", model.score(x_test, y_test))