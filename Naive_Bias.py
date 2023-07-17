import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv(r"C:\Users\ASUS\Desktop\Codes\Python\regression_learning\Social_Network_Ads.csv")

x = df.iloc[:, [1, 2, 3]].values # Extracting the Independent Columns
y = df.iloc[:, 4].values # Extracting the Dependent "purchased" Column

le = LabelEncoder() # Creating the Label Encoder to convert the Gender column into numeric value
x[:, 0] = le.fit_transform(x[:, 0]) # Converting the Gender column into 1 for Male and 0 for Female

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

my_model = GaussianNB() # Creating the GaussianNB instance for Naive Bias Classifying
my_model.fit(x_train, y_train) # Training my Model

pred = my_model.predict(x_test) # Testing my Model
print(x_test)
print(pred)
print(f"Score of Naive Bias model: {my_model.score(x_test, y_test)}")