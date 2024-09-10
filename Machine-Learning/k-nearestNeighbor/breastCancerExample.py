import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

"""
This is a simple exploration into overfitting and underfitting with
KNN. 
"""

cancer=load_breast_cancer()

CANCER_train, CANCER_test, status_train, status_test=train_test_split(
        cancer.data, cancer.target,stratify=cancer.target, random_state=66
    )

def main():
    choose=int(input("(1) data | (2) Plot Training/test model | (3) Test train optimal: "))
    if choose==1:
        data()
    elif choose==2:
        trainAccuracy()
    elif choose==3:
        trainTest()
    else:
        main()

def data():
    print(f"Data Keys: {cancer.keys()}")
    print(f"Data Shape: {cancer['data'].shape}")
    sampleCountDict={n.item(): v.item() for n, v in zip(cancer.target_names, np.bincount(cancer.target))}
    print(f"Sample Counts per Class: {sampleCountDict}")
    
    
    data=[]
    for a, b in np.nditer([cancer.feature_names, cancer.data]):
        case={a.item():b.item()}
        data.append(case)
    data=np.array(data).reshape(569, 30)
    
    print(f"\nFirst 10 sets of data:\n")
    for x in range(10):
        print(f"{x}: {data[x]}\n")
        
def trainAccuracy():
    
    
    train_acc=[]
    test_acc=[]
    
    neighbor_numb=range(1,11)
    
    for neighbor in neighbor_numb:
        model=KNeighborsClassifier(n_neighbors=neighbor)
        model.fit(CANCER_train, status_train)
        train_acc.append(model.score(CANCER_train, status_train))
        test_acc.append(model.score(CANCER_test, status_test))
    plt.plot(neighbor_numb, train_acc, label="Training Accuaracy")
    plt.plot(neighbor_numb, test_acc, label="Testing Accuracy")
    plt.xlabel("n_neighbors")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    
    """
    **This plot shows that with one neighbor, there tends to be overfitting,
    **and our testing accuracy suffers. With 10 neighbors, the model is too simple
    **and we get a case of underfitting. The optimal range appears to be around 6
    **neighbors. However, because this is classifacation, it's perferable to use
    **odd numbers of neighbors, so we can test two instances of 5 and 7 to see
    **which one generalizes the best
    """
    
def trainTest():
    modelFive=KNeighborsClassifier(n_neighbors=5)
    modelSeven=KNeighborsClassifier(n_neighbors=7)
    modelFive.fit(CANCER_train, status_train)
    modelSeven.fit(CANCER_train, status_train)
    targetFive=modelFive.predict(CANCER_test)
    targetSeven=modelSeven.predict(CANCER_test)
    print(f"Test set predictions n=5:{targetFive}")
    print(f"Test set predictions n=7:{targetSeven}")
    
    print(f"Test score model five: {modelFive.score(CANCER_test, status_test):.2f}")
    print(f"Test score model seven: {modelSeven.score(CANCER_test, status_test):.2f}")
    
if __name__=="__main__":
    main()