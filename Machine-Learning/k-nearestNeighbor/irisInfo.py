import numpy as np
from sklearn.datasets import load_iris
irisData=load_iris()

"""
**This file is meant to simply show the iris data set, and various info about 
**the data to be worked with. Given this is generally a first time use of
**machine learning, effort was made to help visualize what pieces of data 
**correspond to target and feature names in the data set
"""

"""
#Basic information on the Iris Data set
print(irisData['DESCR'])
print(f"Target Names:{irisData['target_names']}")
print(f"Feature Names: {irisData['feature_names']}")
"""

#Iris data structure
print(f"Iris Dataset Keys: {irisData.keys()}")
print(f'Type of data: {type(irisData["data"])}')
print(f"Shape of the data: {irisData['data'].shape}")

#This is to show the relation to our data points and their corresponding feature
dataFeatures=[]

for data in range(len(irisData['data'])):
    value=irisData['data'][data]
    names=irisData['feature_names']
    case={
        names[0]:value[0].item(), 
        names[1]:value[1].item(), 
        names[2]:value[2].item(), 
        names[3]:value[3].item()
    }
    dataFeatures.append(case)
print(f"\nFeature names of each flower and their given measurements:")
print(f"----------------------------------------------------------")
for info in dataFeatures[:10]:
    print(info)
    
#The targets and target names
targetNameIndex={
    irisData['target_names'][0].item():'0',
    irisData['target_names'][1].item():'1',
    irisData['target_names'][2].item():'2'
}
print(f"\nTarget Shape: {irisData['target'].shape}")
print(f"Targets: {irisData['target']}")
print("\nA dictionary of all target names, and the corresponding index which they apply to:")
print(targetNameIndex)