"""
The US banking system is a fractional reserve banking system which opperates at a 90:10 ratio.
90% of the money placed into the bank(I.E. a savings account) is leant out. 10% remains on hand
in the bank, for potential withdrawls. Assuming each iteration of leant money goes directly into
another bank, which then lends out the money again and again, until the last possible penny is leant
out, this is an algorithm of what that might look like. 
"""

#this isnt entirely correct, do not merge with main-needs work
import numpy as np

def fractional(deposit):
    leant=[] 
    on_hand=[]
    

    while(deposit>=0.01):
        reserve=deposit*0.1
        deposit-=reserve
        leant.append((deposit))
        on_hand.append(round(reserve, 2))
         
    print(f'Leant:\t{leant:.2f}\n\n')

deposit=10000.00
fractional(deposit)