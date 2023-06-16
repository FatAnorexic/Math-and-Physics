import matplotlib.pyplot as plt
import numpy as np

def f(x):
    if x%2==0:
        return x/2
    else:
        return 3*x+1
def seed():
    seed=float(input("Enter a number: "))
    tree=[seed]
    x=[1]
    i=1
    while f(seed)!=1:

        i+=1
        print(i, '\t',f(seed))
        tree.append(f(seed))
        x.append(i)
        seed=f(seed)
        f(seed)
    plt.plot(x, np.log(tree))

    plt.grid()
    plt.show()
seed()