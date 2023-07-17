import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("C:\\Users\\ASUS\\Desktop\\Codes\\Python\\regression_learning\\insurance_data.csv")

print(df)

sns.lmplot(x="age", y="premium", data=df) # Plotting the graph to see the dataSet

# plt.show()

reg = linear_model.LinearRegression() # Generating the instance of linearRegression model
reg.fit(df[['age']], df['premium']) 
# To train the model i.e. to establish the connection between dependent and independent variables.
# here age is independent and premium is dependent.

print(f"Slope is {reg.coef_} and intercept is {reg.intercept_}")
# In y=mx + c, m is slope and c is intercept.

my_result = reg.predict([[21]])
print(my_result)
