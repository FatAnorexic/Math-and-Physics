import numpy as np
import matplotlib.pyplot as plt
import seaborn
from scipy.optimize import fsolve

seaborn.set_style("darkgrid", {'grid.color':'.6','grid.style':'dots'})
fig=plt.figure()
ax=fig.add_subplot(1,1,1)

def y(x):
    # return 9/5*(1/3*(200+x)+np.sin(x)+(2*np.cos(x))/(200+x)
    #             -(2*np.sin(x))/((200+x)**2))-4600720/((200+x)**2)   #ex_1
    # return (4000-3998*np.exp(-3/800*x) if x<=35.475 else (435.475-x)**2/320) #ex_2
    # return np.piecewise(x, [x<=35.475, x>35.475],[lambda x:4000-3998*np.exp(-3/800*x),
    #                                               lambda x:(435.475-x)**2/320])
    return 112/np.log(3)-1.9468*np.exp(np.log(3)/14*x) #ex_3_final
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

def ex_2():
    '''
    A 1000 gallon holding tank that catches runoff from some chemical process
    initially has 800 gallons of water with 2 ounces of pollution dissolved in
    it. Polluted water flows into the tank at a rate of 3 gal/hr and contains 5
    ounces/gal of pollution in it. A well mixed solution leaves the tank at 3 gal/hr
    as well. When the amount of pollution in the holding tank reaches 500 ounces the
    inflow of polluted water is cut off and fresh water will enter the tank at a
    decreased rate of 2 gal/hr while the outflow is increased to 4 gal/hr. Determine
    the amount of pollution in the tank at any time $t$.
    '''
    print(f't={-800/3*np.log(3500/3998)}')
    x=np.linspace(0,450,100)
    graph_y(x)

def ex_3():
    '''
     A population of insects in a region will grow at a rate that is proportional to
     their current population. In the absence of any outside factors the population
     will triple in two weeks time. On any given day there is a net migration into
     the area of 15 insects and 16 are eaten by the local bird population and 7 die of
     natural causes. If there are initially 100 insects in the area will the population
     survive? If not, when do they die out?
    '''
    # print(f'c={100-112/np.log(3)}')
    # print(f't={(14*np.log(112/(1.9468*np.log(3))))/np.log(3)}')
    x=np.linspace(0, 50, 100)
    graph_y(x)

# def graph_lim(x):
    #
def graph_y(x):
    ax.plot(x, y(x), color='blue', label=r'$P(t)=\frac{112}{\ln3}-1.9468e^{\frac{\ln(3)}{14}t}$')
    ax.spines[['left', 'bottom']].set_position('zero')
    ax.set_xticks(range(0,55,10))
    ax.set_yticks(range(0,120,20))
    ax.set_ylabel(ylabel='P(t)', loc='top', rotation=0)
    ax.set_xlabel(xlabel='t',loc='right')
    plt.legend(loc='best', prop={'size': 10})
    plt.show()

if __name__=='__main__':
    # ex_1()
    # ex_2()
    ex_3()