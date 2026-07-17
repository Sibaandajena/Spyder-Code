# BACKED
# IMPORT LIBRARY
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# IMPORT THE DATASET
dataset= pd.read_csv(r"C:\Users\HP\Desktop\ALL DATA SET\Salary_Data.csv")

# INDEPENDENT VARIABLE
X = dataset.iloc[:, :-1]
# DEPENDENT VARIABLE
Y = dataset.iloc[:,-1]

# SPLIT THE DATA
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
   X, Y, test_size=0.20, random_state=0)



from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

Y_pred = regressor.predict(X_test)


plt.scatter(X_test, Y_test, color='Red')  # Real Salary data(testing)
plt.plot(X_train, regressor.predict(X_train), color = 'blue')   
plt.title('Salary of employee based on experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()


m_coef = regressor.coef_
print(m_coef)

c_intercept = regressor.intercept_
print(c_intercept)

y_12 = m_coef*12 + c_intercept
print(y_12)


bias_score = regressor.score(X_train, Y_train)
print(bias_score)

variance= regressor.score(X_test, Y_test)
print(variance)



dataset.mean()
dataset['Salary'].mean()
dataset['YearsExperience'].mean() 

dataset.median()
dataset['Salary'].median()
dataset['YearsExperience'].median()

dataset.var()
dataset['Salary'].var()
dataset['YearsExperience'].var()

dataset.std()   # standard daviation
dataset['Salary'].std()
dataset['YearsExperience'].std() 


from scipy.stats import variation
variation(dataset.values)
variation(dataset['Salary'])
variation(dataset['YearsExperience'])

dataset.corr()  # corealation

dataset['Salary'].corr(dataset['YearsExperience'])


# SKEWNWSS 

dataset['Salary'].skew()

dataset.sem()

# Z-score 
#

import scipy.stats as stats

dataset.apply(stats.zscore) # this will give Z-score of entire dataframe

stats.zscore(dataset['Salary'])


# ANOVA # SSR, SSE, SST

Y_mean = np.mean(Y)
SSR = np.sum((Y_pred-Y_mean)**2)
print(SSR)


Y=Y[0:6]
SSE = np.sum((Y-Y_pred)**2)
print(SSE)

mean_total = np.mean(dataset.values)
SST = np.sum((dataset.values-mean_total)**2)
print(SST)


r_square = 1- (SSR / SST)
r_square 


print(r_square)
print()







