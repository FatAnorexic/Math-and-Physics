from functools import lru_cache

def main():

    for i in range(1,400):
        print(fib(i))

#This is one of the chache decorators->we're using the last 5 values in memory
#By doing so, we greatly improve the speed at which our code executes.

@lru_cache(maxsize=5)
def fib(n):
    if n==0:
        return 0
    elif n==1:
        return 1
    else:
        return fib(n-1)+fib(n-2)
main()