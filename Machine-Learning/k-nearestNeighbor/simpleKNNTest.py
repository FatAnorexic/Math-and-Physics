import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

irisData=load_iris()
knn=KNeighborsClassifier(n_neighbors=1)


"""
**The train test split that's been imported will cut and split the 
**data for us, removing the need to manually carve our total data 
**set. 
"""

"""
**Most conventions would label things X_train/X_test, y_train/y_test.
**As I've progressed in my understanding of code, I've come to learn this
**can be extremely confusing for the reader, and adds unecessary mental 
**taxation. The 'X' is capitalized as it is a matrix(see any book on linear algebra)
**and 'y' is a 1d vector pointing to something. 
"""
DATA_train, DATA_test, target_train, target_test=train_test_split(
    irisData['data'], irisData['target'], random_state=0
    )

def main():
    choose=int(input("(1) Plot Data (2) Train model (3) Test model: "))
    if choose==1:
        dataPlotting()
    elif choose==2:
        training()
    elif choose==3:
        predictions()
    else:
        main()      

def dataPlotting():
    #Check the split data and their ratios
    print(len(DATA_test)/len(DATA_train))

    #Create a scatter matrix using pandas to visualize data for inconsistencies
    #and visual aid. Sometimes ML is not needed to solve the problem

    irisFrame=pd.DataFrame(DATA_train, columns=irisData.feature_names)
    pd.plotting.scatter_matrix(
        irisFrame, c=target_train, figsize=(20,20), marker='o', hist_kwds={'bins':20},
        s=60, grid='true')
    plt.show()
    
    """
    **The three species of plants seem to be fairly well separated. Meaning an ML model
    **will likely be able to learn how to distinguish them. Of special note here, however,
    **this is an example case, and as such the data 'fits' very snuggly into an ML model
    """

"""
Trains the model and saves weights to a binary pickle file. Saving seems unecessary now
but in later models it'll be easier to separate training and testing. Best to get used 
to saving weights now.
"""
def training():
    knn.fit(DATA_train, target_train)
    with open('iris_model.pkl', 'wb') as f:
        pickle.dump(knn, f)

def predictions():
    with open('iris_model.pkl', 'rb') as f:
        knn_new=pickle.load(f)
    target_pred=knn_new.predict(DATA_test)
    print(f"test set predictions: {target_pred}")
    print(f"Test set score: {np.mean(target_pred==target_test):.2f}")
    print(f"Test set score: {knn_new.score(DATA_test, target_test):.2f}")

if __name__=="__main__":
    main()