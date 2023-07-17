import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

data = {
    'Position': ['Manager', 'Senior Developer', 'Analyst', 'Sales Representative', 'Intern', 'Director', 'Consultant', 'Associate', 'Administrator', 'Engineer'],
    'Level': [1,2,3,4,5,6,7,8,9,10],
    'Salary': [10000,20000,40000,60000,80000,95000,100000,150000,200000,300000]
}

df = pd.DataFrame(data)
# print(df)

x = df.iloc[:, 1:2].values # To extract all "Level" from DtataFrame in an Array. [:, 1:2] means all rows and 1st column.
y = df.iloc[:,2].values # To extract all "Salary" from DtataFrame in an Array.

sns.lmplot(x='Level', y='Salary', data=df)

plt.show()

reg = linear_model.LinearRegression()
reg.fit(x, y) # Use this syntax if just using columns X and Y.
my_pred = reg.predict([[6.5]])
print(f"By Linear Regression, Value of 6.5 is: {my_pred}")

# POLYNOMIAL REGRESSION
'''
Create an instance of the PolynomialFeatures class from the preprocessing module. 
This object will be used to transform the input features into polynomial features of the specified degree (in this case, degree 2).
'''
poly = PolynomialFeatures(degree=2) # Create a polynomial features object of degree 2
x_poly = poly.fit_transform(x) # Transform the input features into polynomial features

polyReg = linear_model.LinearRegression() # Create a linear regression object for polynomial regression
polyReg.fit(x_poly, y) # Transform the input features into polynomial features

poly_Predict = polyReg.predict(poly.fit_transform([[6.5]])) 
# Transform the input value of 6.5 into polynomial features and use the trained polynomial regression model (polyReg) to make a prediction. 

print(f"By Polinomial Regression, Value of 6.5 is: {poly_Predict}")