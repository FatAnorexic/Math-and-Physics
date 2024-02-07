import matplotlib.pyplot as plt
import numpy as np
import seaborn

seaborn.set(style='darkgrid')

def y(x):
    # place your final solution to the equation here
    return -16/(x**4*limit(x)) #ex1
def limit(x):
    # place the limiting interval to solve for here
    return (1+16*np.log(x/2)) #ex1
def ex_1():
    #set the interval of validity and number of iterations you wish to compute
    x=np.linspace(2*np.exp(-1/16)+0.01, 5, 100)
    #set a resonable interval you wish to check the limit of x
    x_lim=np.linspace(-10,10,100)

    # graph_limit(x_lim)
    graph_y(x)

def graph_limit(x):
    # graph the limiting factor here. Often used to get an idea
    #for a range in which to use fsolve or other methods in finding roots
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(x, limit(x), color='red', label=r'$$')

    ax.spines[['left', 'bottom']].set_position(('data', 0))
    ax.spines[['right', 'top']].set_visible(False)
    plt.legend(loc='best', prop={'size': 6})
    plt.show()

def graph_y(x):
    # Graph your final solution here

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(x, y(x), color='blue',
            label=r'$y(x)=-\frac{16}{x^4(1+16\ln(\frac{x}{2}))}$')

    ax.spines[['left', 'bottom']].set_position('zero')
    ax.spines[['right', 'top']].set_visible(False)
    # ax.set_aspect('equal')
    seaborn.despine(ax=ax, offset=0)
    plt.legend(loc='best', prop={'size': 10})
    plt.show()


if __name__=='__main__':
    #call your main function here
    ex_1()