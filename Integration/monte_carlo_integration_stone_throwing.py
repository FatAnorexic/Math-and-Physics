import random
import numpy as np
import matplotlib.pyplot as plt

"""
This uses random tossing of pebbles to estimate the size of a pool of size π.
One is a much more refined optimal solution f() where as the other is one the 
book Computational Physics by Rubin Landau, Manuel Paez, and Cristian Bordeianu
walks you through 
"""

def f():
    N=100000
    npond=0
    for i in range(1, N):
        d=np.hypot(random.random(),random.random())
        if d<1:npond+=1
        Apond=(4*npond/i)
    print('{0:.3f}'.format(Apond))




def f_book():
    N=100000
    npond=0
    nbox=0
    #The lists are merely for the graph, we could add a similar set of lists to f() 
    # for a comparison plot.
    X_pond=[]
    Y_pond=[]
    X_box=[]
    Y_box=[]
    for i in range(1, N):
        x=random.uniform(-1, 1)
        y=random.uniform(-1,1)
        if np.sqrt(x**2+y**2)<1:npond+=1;X_pond.append(x);Y_pond.append(y)
        else:nbox+=1;X_box.append(x);Y_box.append(y)
        Apond=4*npond/(npond+nbox)
    plt.scatter(X_box,Y_box, marker='D', color='green')
    plt.scatter(X_pond,Y_pond, marker='o', color='blue')
    plt.grid()
    plt.show()
    print('{0:.3f}'.format(Apond))

def main():
    f()
    f_book()

if __name__=='__main__':
    main()