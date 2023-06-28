from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

can=load_breast_cancer()

Xtrain, Xtest, ytrain, ytest= train_test_split(can.data, can.target, random_state=1)

print(f'Training shape {Xtrain.shape}')
print(f'Test shape {Xtest.shape}')