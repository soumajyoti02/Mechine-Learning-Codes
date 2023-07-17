import pandas as pd
from sklearn import linear_model

df = pd.read_csv("C:\\Users\\ASUS\\Desktop\\Codes\\Python\\regression_learning\\ins_multiple.csv")

print(df)

mean_Value = df.height.mean()
df["height"] = df["height"].fillna(mean_Value) # There is a null value in height. So modifying dataFrame
print(df)

reg = linear_model.LinearRegression() # Making the linearRegression instance

reg.fit(df[['age', 'height', 'weight']], df['premium']) # Training the model. Pass all the dependent variables inside.

print(f"Coeffecients are {reg.coef_} and intercept is {reg.intercept_}")

pred_value = reg.predict([[27, 167.56, 60]]) # Making Prediction

print(pred_value)