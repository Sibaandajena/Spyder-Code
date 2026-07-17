# IMPORT LIBRARY
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# IMPORT THE DATASET
dataset= pd.read_csv(r"C:\Users\HP\Desktop\ALL DATA SET\Data.csv") 

# INDEPENDENT VARIABLE
X = dataset.iloc[:, :-1].values
# DEPENDENT VARIABLE
Y = dataset.iloc[:,3].values

# SKLEARN FILL MISSING NUMERICAL VALUE
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent')

imputer = imputer.fit(X[:,1:3])  # n-1 (3-1=2)

X[:, 1:3]= imputer.transform(X[:,1:3])

# IMPUTE CATEGORICAL VALUE FOR DEPENDENT 
from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder()

labelencoder_X.fit_transform(X[:,0])

X[:,0] = labelencoder_X.fit_transform(X[:,0])


# IMPUTE CATAGORICAL VALUE FOR DEPENDENT
labelencoder_Y = LabelEncoder()
Y =labelencoder_Y.fit_transform(Y)

# SPLIT THE DATA
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
   X, Y, test_size=0.3, random_state=0)





