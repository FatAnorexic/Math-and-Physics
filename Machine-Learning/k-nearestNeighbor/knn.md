# K-Nearest Neighbor
This is a generalization of the algorithm and an overview of what it aims to do. It belongs to the supervised learning domain of machine learning, and is often used in classification. It is often a first when stepping into the world of ML, and is fairly easy to implement. Though a draw back is that it does not scale well when larger data sets are introduced due to its nature of storing this data in memory, and when the dimensionality of data points are too high can be prone to overfitting. 

# The Minkowski Distance

$$d(x,y)=\Sigma^{n}_{i=1}(x_i-y_i)^p)^{\frac{1}{p}}$$
KNN uses a distance metric for identifying the nearest neighbors to a given point. While it can be the **Euclidean Distance** or the **Manhattan Distance**, both of these can be represented by changing the values of p with the **Minkowski Distance**. if p=1, it will become the Manhattan Distance and if p=2, Euclidean.

# Choosing the right K-value
There are a two rules of thumb that go into this:
1. K-values should be odd, not even, in cases of classification to avoid ties. 
2. If your data has a lot of outliers, a higher K-value should be chosen.

# Algorithm Example

```python
import numpy as np
from collections import Counter
from heapq import nsmallest

class KNN:
	def __init__(trainData, trainTarget,labels):
		self.data=zip(trainData,trainTarget)
		self.labels=labels

	def classify(self, prediction_point, k: int=3): #k default set to 3
		distances=(
		(self.eucldean_distance(point[0], prediction_point), point[0])
		for point in self.data
		)

		votes=(vote[i] for vote in nsmallest(k, distances))

		result=Counter(votes).most_common(1)[0][0]
		return self.labels[result]

	@staticmethod
	def euclidean_distance(a, b):
		return np.linalg.norm(a-b)
```
[^1]: source: The Algorithms/Python : tianyizheng02 and poyea
