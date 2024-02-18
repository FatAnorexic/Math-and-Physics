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
    # return 112/np.log(3)-1.9468*np.exp(np.log(3)/14*x) #ex_3_final
    return 98-108*np.exp(-x/10) #ex_4_v(t)

def interval(x):

    #we subtract the 100 from the left side in order to get the zero of the function
    #This allows fsolve to solve for x properly
    #return 98 * x + 1080 * np.exp(-1 / 10 * x) - 1080  # ex_4_distance_s(t)_graphing
    return 98 * x + 1080 * np.exp(-1 / 10 * x) - 1180  # ex_4_distance_s(t)
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

def ex_4():
    """
    A 50 kg object is shot from a cannon straight up with an initial
    velocity of 10m/s off a bridge that is 100 meters above the ground.
    If air resistance is given by 5v determine the velocity of the mass
    when it hits the ground.
    """
    x_t=np.linspace(0,8,100)
    # graph_lim(x_t)
    print(f't={fsolve(interval,[-3,5])}')
    v_t=np.linspace(0,5.98,100)
    print(f'v(5.981)={y(5.981)} m/s')
    graph_y(v_t)
def ex_5():
    """
    A 50 kg object is shot from a cannon straight up with an initial velocity of 10m/s off a bridge that is 100 meters
    above the ground. If air resistance is given by 5v2 determine the velocity of the mass at any time t.
    """
    #we need to find time t when the initial velocity given=0 and the object is at its zenith
    print(f'0=v(t) -> t={-10/np.sqrt(98)*np.arctan(-10/np.sqrt(98))}')


def graph_lim(x):
    ax.plot(x, interval(x), color='red', label=r'$100=98t+1080e^{-\frac{1}{10}t}-1080$')
    ax.spines[['left', 'bottom']].set_position(('data', 0))
    ax.set_xticks(range(0, 9, 1))
    # ax.set_yticks(range(0, 120, 10))
    ax.set_ylabel(ylabel='s(t)', loc='top', rotation=0)
    ax.set_xlabel(xlabel='t', loc='right')
    plt.legend(loc='best', prop={'size': 10})
    plt.show()
def graph_y(x):
    ax.plot(x, y(x), color='blue', label=r'$v(t)=98-108e^{-\frac{1}{10}t}$')
    ax.spines[['left', 'bottom']].set_position('zero')
    ax.set_xticks(range(0,7,1))
    ax.set_yticks(range(-10,50,5))
    ax.set_ylabel(ylabel='v(t) m/s', loc='top', rotation=0)
    ax.set_xlabel(xlabel='t',loc='right')
    ax.xaxis.set_label_coords(1.05,0.20)
    plt.legend(loc='best', prop={'size': 10})
    plt.show()

if __name__=='__main__':
    # ex_1()
    # ex_2()
    # ex_3()
    # ex_4()
    ex_5()