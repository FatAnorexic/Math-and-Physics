import matplotlib.pyplot as plt
import numpy as np
import seaborn
import pandas as pd

seaborn.set_style("darkgrid", {'grid.color':'.6','grid.style':'dots'})

def y(x):
    # place your final solution to the equation here
    # return -16/(x**4*limit(x)) #ex1
    # return np.power(((139*np.exp(15*x)-3*np.exp(-2*x))/17), 0.333) #ex2
    # return -(2/np.power(limit(x),1/3)) #ex 3
    return (x**3-2*x**(3/2)+1)/(9*x) #ex4
def limit(x):
    # place the limiting interval to solve for here
    # return (1+16*np.log(x/2)) #ex1
    # return 4*x-4+5*np.exp(-x) #ex 3
    return 9*x #ex 4
def ex_1():
    #set the interval of validity and number of iterations you wish to compute
    x=np.linspace(2*np.exp(-10), 5, 100)
    #set a resonable interval you wish to check the limit of x
    x_lim=np.linspace(-10,10,100)

    # graph_limit(x_lim)
    graph_y(x)

def ex_2():
    x = np.linspace(-0.2, 0.4, 10000)
    graph_y(x)

def ex_3():
    xlim=np.linspace(-100,100,100)
    graph_limit(xlim)

    # x=np.linspace(-12,12,1000)
    # graph_y(x)
def ex_4():
    # xlim=np.linspace(-10,10,100)
    # graph_limit(xlim)

    x=np.linspace(0.0`1,10,100)
    graph_y(x)
def graph_limit(x):
    # graph the limiting factor here. Often used to get an idea
    #for a range in which to use fsolve or other methods in finding roots
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(x, limit(x), color='red', label=r'$4x-4+5e^x$')

    ax.spines[['left', 'bottom']].set_position(('data', 0))
    ax.spines[['right', 'top']].set_visible(False)
    plt.legend(loc='best', prop={'size': 6})
    plt.show()

def graph_y(x):
    # Graph your final solution here

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(x, y(x), color='red',
            label=r'$y(x)=y(x)=\frac{x^3-2x^{\frac {3} {2}}+1}{9x}$')

    # ax.spines[['left', 'bottom']].set_position('zero')
    # ax.spines[['right', 'top']].set_visible(False)
    # ax.set_aspect('equal')
    seaborn.despine(ax=ax, offset=0)
    plt.legend(loc='best', prop={'size': 10})
    plt.show()


if __name__=='__main__':
    #call your main function here
    # ex_1()
    #ex_2()
    # ex_3()
    ex_4()