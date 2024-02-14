import numpy as np
import matplotlib.pyplot as plt
import seaborn
from scipy.optimize import fsolve

seaborn.set_style("darkgrid", {'grid.color':'.6','grid.style':'dots'})
fig=plt.figure()
ax=fig.add_subplot(1,1,1)

def y(x):
    return 9/5*(1/3*(200+x)+np.sin(x)+(2*np.cos(x))/(200+x)
                -(2*np.sin(x))/((200+x)**2))-4600720/((200+x)**2)   #ex_1
# def interval(x):

def ex_1():
    """
    A 1500 gallon tank initially contains 600 gallons of water with 5 lbs
    of salt dissolved in it. Water enters the tank at a rate of 9 gal/hr
    and the water entering the tank has a salt concentration of r$15(1+cos(t))$
    lbs/gal. If a well mixed solution leaves the tank at a rate of 6 gal/hr,
    how much salt is in the tank when it overflows?
    """

    print(f't={y(300)}')

    x=np.linspace(0,350,1000)
    graph_y(x)

# def graph_lim(x):
    #
def graph_y(x):
    ax.plot(x, y(x), color='blue', label=r'$Q(t)=\frac{big}{ass}eqation$')
    ax.spines[['left', 'bottom']].set_position('zero')
    ax.set_xticks(range(0,350,50))
    plt.legend(loc='best', prop={'size': 10})
    plt.show()

if __name__=='__main__':
    ex_1()