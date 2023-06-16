#This is an example of newton-raphson searching using central difference
import numpy as np

x=4.0;  dx=3.0e-1; eps=0.2
imax=100
def f(x):
    return 2*np.cos(x)-x

for i in range(0, imax+1):
    F=f(x)
    if(abs(F)<=eps):print('\n Root found, F=',F,' tolerance eps=',eps);break
    print('Iteration# =', i, 'x=', x, 'f(x)=',F)
    df=(f(x+dx/2)-f(x-dx/2))/dx
    dx= -F/df
    x+=dx
#for purposes in chapter 6, we would use something like forward difference,
# rather than central, due to computing limits
