from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

can=load_breast_cancer()
scaler=MinMaxScaler()

Xtrain, Xtest, ytrain, ytest= train_test_split(can.data, can.target, random_state=1)

scaler.fit(Xtrain)

#print(f'Training shape {Xtrain.shape}')
#print(f'Test shape {Xtest.shape}')

#transform the data
xtrain_scaled=scaler.transform(Xtrain)
xtest_scale=scaler.transform(Xtest)
#This is what the transformation will do-essentially prepossessing our data to 
#a range between 0 or 1-This is not true for the test set however
"""
print(f'Transformed shape {xtrain_scaled.shape}')
print(f'per-feature minimum before scale:\n{Xtrain.min(axis=0)}')
print(f'per-feature maximum before scale:\n{Xtrain.max(axis=0)}')
print(f'per-feature min after:\n{xtrain_scaled.min(axis=0)}')
print(f'per-feature max after:\n{xtrain_scaled.max(axis=0)}')

#Rather than 0 or 1 our bounds can be slightly below zero or a bit above 1  
print(f'per-feature min after:\n{xtest_scale.min(axis=0)}')
print(f'per-feature max after:\n{xtest_scale.max(axis=0)}')

# This is because the MinMaxScalar always applies the exact same transormation to
#the training and test set. Meaning the transform method always subtracts from the 
#training set minimum and divides by the training range. Which may be different minimums 
#and range for the test set.We must fix this. 
"""
