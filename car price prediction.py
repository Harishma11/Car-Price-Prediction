import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#load data
d=pd.read_csv("C:/Users/Harishma/OneDrive/Desktop/carprice/CarPrice.csv")
df=pd.DataFrame(d)
print(df)

#first 5 rows
print(df.head())


#last 5 rows
print(df.tail())

#size
print(df.shape)

#describe
print(df.info())

#check null rows
print(df.isnull().sum())


print(df['CarName'].value_counts())

#independent and dependent variable
X=df[['symboling','price']]
Y=df['price']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#linear regression
lm = LinearRegression()
lm.fit(X_train, Y_train)
LinearRegression()

#predictions
predictions = lm.predict(X_test)

#barplot
sns.barplot(x='symboling',y='price',data=df)
plt.show()

#boxplot
sns.boxplot(x=df['symboling'],y=df['price'],palette=("coolwarm_r"))
plt.show()

sns.boxplot(df['price'])
plt.show()

#displot
sns.displot(df['aspiration'])
plt.show()

sns.displot(df['price'])
plt.show()

sns.displot(df['symboling'])
plt.show()

#scatterplot

sns.scatterplot(x=df['fueltype'],y=df['price'],hue=df['drivewheel'],palette=("plasma"))
plt.xlabel("Fuel")
plt.ylabel("Price")
plt.show()










