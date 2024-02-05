import numpy as np


def f(x):
    return -4*np.power(x,3)+12*np.power(x,2)+9

def bisection(a, b, err):
    while(np.abs(a-b)>=err):
        c=(a+b)/2.0
        if f(a)*f(c)>err:
            a=c
        elif f(a)*f(c)<err:
            b=c

    return c
def main():
    test=bisection(-3,4,1e-8)
    print(f'x= [{test}]')

if __name__=='__main__':
    main()