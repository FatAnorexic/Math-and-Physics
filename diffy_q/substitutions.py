import matplotlib.pyplot as plt
import numpy as np
import seaborn
from scipy.optimize import fsolve

seaborn.set_style("darkgrid", {'grid.color':'.6','grid.style':'dots'})
fig=plt.figure()
ax=fig.add_subplot(1,1,1)

def y(x):
    return -1*np.sqrt(limit(x)/(x**2)) #ex_1_final
def limit(x):
    return 228-2*x**4   #ex1_limit
def eq_1():
    # xlim=np.linspace(-3,3,100)
    # graph_limit(xlim)

    # print(f'x={fsolve(limit,[-3.5,3.5])}')
    # x=[-3.26757988  3.26757988]

    x=np.linspace(0.1,3.25,100)
    print(y(x))
    graph_y(x)

def graph_limit(x):

    ax.plot(x,limit(x),color='red', label=r'$228-4x^4\geq0$')
    seaborn.despine(ax=ax, offset=0)
    plt.legend(loc='best', prop={'size': 10})
    plt.show()
def graph_y(x):

    ax.plot(x, y(x), color='blue', label=r'$y(x)=-\sqrt{\frac{228-2x^4}{x^2}}$')
    seaborn.despine(ax=ax, offset=0)
    ax.set_xticks(range(5))
    plt.legend(loc='best', prop={'size': 10})
    plt.show()

if __name__=='__main__':
    eq_1()