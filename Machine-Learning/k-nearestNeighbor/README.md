# Description
These files demonstrate the basic [K-Nearest Neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm#cite_note-:1-2) use case, and a simple overview of what it is. First created in 1951 By **Evelyn Fix** and **Joseph Hodges** [Discriminatory Analysis-Nonparametric Discrimination: Consistency Properties](https://apps.dtic.mil/sti/citations/ADA800276), and later expanded by **Thomas Cover** and **Peter Hart** in their paper [Nearest neighbor pattern classification](https://ieeexplore.ieee.org/document/1053964/authors#authors)

---

# Dependencies

The following are a list of dependent packages you'll need to install into python in order for these files to work. 
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`

A simple run of the command line in your console should be able to grab all necessary files
`pip install numpy pandas scikit-learn matplotlib`

If you're working in a virtual environment, you'll need to ensure these packages are linked to that as well. 

if you wish to exclude any weights saved to execution, be sure to include `*.pkl` to your `.gitignore` file

---

# What it does

irisInfo.py grabs the iris dataset from scikit, and logs that data to the console. Recommended if it is your first time looking at this. 

simpleKNNTest.py will plot the data, train the model using the iris dataset and test that trained model. 

breastCancerExample.py demonstrates plotting training and testing data using knn, and determining the best 'generalization' for n-neighbors.