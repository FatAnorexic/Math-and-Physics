import numpy as np
import matplotlib.pyplot as plt
import deriv_func1

#Forwad Diff = (y(t+h)-y(t))/h
#Central Diff = (y(t+h/2)-y(t-h/2))/h

def main():
    t = [0.1, 1.0, 100.0]
    h = 100
    print('Enter the derivative you wish to calculate: \n(1)First order\n(2)Second order\n')
    choice=int(input(': '))
    if choice==1:
        print('What type of method would you use?\n(1)Forward Difference\n(2)Central Difference\n(3)Extrapolated Difference:\n')
        choice2=int(input(': '))
        if choice2==1:
            diff=deriv_func1.Forward(np.cos, t[2], h)
            out=diff.get_forward_diff()
            output(out)


def output(out):

    t = [0.1, 1.0, 100.0]
    print('dcos(x)/dt')
    for j in range(3):
        h = 100
        i = 1
        ERROR_FD=[]
        Big_H=[]
        while h>=np.finfo(np.float32).eps:
            output=out
            err_fd = (np.finfo(float).eps) / h
            print(i, '\t\t', output, '\t\t', format(-np.cos(t[j]), '0.16f'), '\t\t', h, '\t\t', t[j], '\t\t', err_fd)
            ERROR_FD.append(err_fd)
            Big_H.append(h)
            h=h/10
            i+=1

            if out == -np.sin(t[j]):
                print(out, '\t\t', h,'\t\t',t[j])
                print("Most accurate!")
                print()

        plt.plot(np.log10(ERROR_FD), np.log10(Big_H))
        plt.xlabel('Log_10 Error')
        plt.ylabel('Log_10 h')
        plt.grid()
        plt.show()
        print()


    print('de^x/dx')
    for j in range(3):
        h = 100
        i = 1
        ERROR_FD2 = []
        Big_H2 = []
        while h>=np.finfo(np.float32).eps:
            FD2=deriv_func1.Second_Derriv(np.exp, t[j], h)
            output=FD2.get_forward_diff()
            err_fd2 = (np.finfo(float).eps) / h
            print(i, '\t\t', format(output, '.16f'), '\t\t', format(np.exp(t[j]), '0.16f'), '\t\t', h, '\t\t', t[j], '\t\t', err_fd2)
            ERROR_FD2.append(err_fd2)
            Big_H2.append(h)
            h=h/10
            i+=1

            if FD2 == np.exp(t[j]):
                print(FD2, '\t\t', h, '\t\t', t[j])
                print("Most accurate!")
                print()
        plt.plot(np.log10(ERROR_FD2), np.log10(Big_H2))
        plt.xlabel('Log_10 Error')
        plt.ylabel('Log_10 h')
        plt.grid()
        plt.show()
        print()
#output()

def deriv():
    fd=deriv_func1.Forward(np.cos, t=0.00, h=1e-8)
    output=fd.get_extrapolated()
    print(output)
#deriv()
def output2nd():

    t=0.1
    h=np.pi/10

    while h>np.finfo(float).eps:
        second=deriv_func1.Second_Derriv(np.cos, t, h)
        second_extend=deriv_func1.Second_Derriv(np.cos, t, h)
        output=second.get_second_deriv()
        output_extend=second_extend.second_deriv_expanded()
        print(output, '\t\t', output_extend, '\t\t', -np.cos(t))
        h=h/10
#output2nd()



"""
def func(x):
    return x**2


def momentum_force():
    m=1
    x = 100.0
    h = 10.0
    fun=func(x)
    print(fun)
    i=1
    while h>np.finfo(float).eps:
        ve=deriv_func1.Forward(fun,x, h)
        velocity=ve.get_forward_diff()
        momentum=m*velocity

        acc=deriv_func1.Second_Derriv(fun, x, h)
        acceleration=acc.get_second_deriv()
        force=m*acceleration
        print(i, '\t\t', momentum, '\t\t', force, '\t\t', h)
        h=h/10
        i+=1
momentum_force()"""

#main()