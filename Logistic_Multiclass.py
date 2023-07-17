import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\ASUS\Desktop\Codes\Python\regression_learning\iris.csv")
df["species"].replace({'setosa' : 1, 'versicolor' : 2, 'virginica' : 3}, inplace=True)

x_train, x_test, y_train, y_test = train_test_split(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']], df['species'], test_size=0.2) 
# This will devide the DataFrame into 80% train data and 20% test data randomly.

print(x_test)

reg = LogisticRegression() # Generate a Logistic Regression instance.
reg.fit(x_train, y_train) # Training the model.
print(reg.predict(x_test)) # Generating the prediction by x_test set.

print(reg.score(x_test, y_test)) # Checking the score of model i.e. how good the model is.

sns.pairplot(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']], hue='species')
plt.show()