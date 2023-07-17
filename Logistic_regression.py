import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = {
    "age": [21, 48, 32, 41, 20, 35, 20, 23],
    "brought": ["no", "yes", "yes", "yes", "no", "yes", "no", "no"]
}
df = pd.DataFrame(data)
df["brought"].replace({'no': 0, 'yes': 1}, inplace=True)

# This will devide the DataFrame into 80% train data and 20% test data randomly.
x_train, x_test, y_train, y_test = train_test_split(df[['age']], df["brought"], test_size=0.2) 
print(x_test)

reg = LogisticRegression() # Defining the logistic Regression instance

reg.fit(x_train, y_train) # Training the model

test = reg.predict(x_test) # Testing the model by using the splitted test Data
print(test)

print(reg.predict([[26]]))
print(reg.score(x_test, y_test))