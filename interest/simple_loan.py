import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def future_value(principle, interest, years):
    return principle*(1+interest)**years

def average_depreceation(current, initial, time):
    return (current-initial)/time
def deprec_ratio(depreciation, init):
    return depreciation/init*100

def bank_take():
    i=float(input("Enter the APR of the loan: "))
    t=int(input("Enter the number of years of the loan: "))
    p=float(input("Enter the principle of the loan amount: "))
    total_bank_take=future_value(p, i, t)
    print(f"Bank makes: {total_bank_take:.2f}")
    
def loss_ratio():
    current_val=float(input("Enter the amount of the car being offered: "))
    init_cost=float(input("Enter the initial price of the car: "))
    time=int(input("How many years have you had the car: "))
    d_amount=average_depreceation(current_val, init_cost, time)
    depreciation=deprec_ratio(d_amount, init_cost)
    
    print(f"Average Depreciation: {depreciation:.2f}%")

def main():
    choice=int(input("Enter (1) for total amount you pay to bank, (2) for the average loss on your vehicle: "))
    if choice==1: bank_take()
    elif choice==2: loss_ratio()
    else: main()
    
if __name__=="__main__":
    main()