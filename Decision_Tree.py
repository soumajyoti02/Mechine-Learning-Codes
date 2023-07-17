import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

data = {
    'outlook': ['sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy', 'overcast', 'sunny', 'sunny', 'rainy', 'sunny', 
                'overcast', 'overcast', 'rainy', 'overcast', 'sunny', 'sunny', 'rainy', 'sunny', 'overcast'],
    
    'humidity': ['high', 'high', 'high', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'normal', 'normal', 
                    'high', 'normal', 'high', 'normal', 'high', 'normal', 'high', 'normal', 'high'],
    
    'windy': ['strong', 'strong', 'weak', 'weak', 'weak', 'strong', 'strong', 'weak', 'weak', 'weak', 'strong', 'strong', 
                'weak',   'strong', 'weak', 'weak', 'strong', 'strong', 'weak', 'strong'],
    
    'play': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 
                'yes', 'yes', 'no', 'yes']
}
df = pd.DataFrame(data)

# Converting the Text columns into numbers to perform operations.
outlook = LabelEncoder()
humidity = LabelEncoder()
windy = LabelEncoder()
play = LabelEncoder()

df["outlook"] = outlook.fit_transform(data['outlook'])
df["humidity"] = humidity.fit_transform(data['humidity'])
df["windy"] = windy.fit_transform(data['windy'])
df["play"] = play.fit_transform(data['play'])

features_cols = ['outlook', 'humidity', 'windy'] 
x = df[features_cols] # To extract only these columns from the dataFrame
y = df['play']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) 
# This will devide the DataFrame into 80% train data and 20% test data randomly.

classifier = DecisionTreeClassifier(criterion='gini') # Generating a Decision Tree classifier
classifier.fit(x_train, y_train) # Training that clasifier.
my_pred = classifier.predict(x_test) # making prediction using that classifier.

print(x_test)
print(my_pred) # Viewing my prediction
print("Decision Tree Score: ", classifier.score(x_test, y_test)) # Checking the model's SCORE

tree.plot_tree(classifier) # Plotting the Decision Tree Decision
plt.show()