from functools import lru_cache

def main():

    for i in range(1,400):
        print(fib(i))

#This is one of the cache decorators->we're using the last 3 values in memory
#By doing so, we greatly improve the speed at which our code executes.
#There is a limit to how big n can be. The recursion depth limit is default
#at 1000. This can be increased by sys.setrecursionlimit(n), but is highly not
#reccomended.

@lru_cache(maxsize=3)
def fib(n):
    if n==0:
        return 0
    elif n==1:
        return 1
    else:
        return fib(n-1)+fib(n-2)


if __name__=="__main__":
    import doctest
    doctest.testmod()
    main()
