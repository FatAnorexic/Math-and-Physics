/*
What is multiplacation? We all remember learning our times tables, and certainly we probably remember
learning the first instances of long hand multiplacation. No doubt some of you may have deduced or
been taught certain tricks such as 120*100 is simply->12*1000= 12000->connect all the zeros.
But if you had to describe what multiplactation to a todler how would you do so?

Often we just brute force the problem in education, and start with simple 2*2, 1*2, 2*3, etcetera.
However, what if the child were incapable of memorizing such tables? What if the logical leap just
couldn't be overcome? What if this child were a computer you were trying to teach? One incapable of
having a multiplacation operator?

Answer: summation. All multiplacation is the sum of one number n added together, m number of times.
That is to say a*b=a[1]+a[2]+a[3]+...+a[b]. Below is such an algorithm written in c++. A python version
would look something like (for n in range(m):...)*/

#include <iostream>
#include <cstdlib>

int multiplacation(int n, int m) {
    int sum = 0;
    for (int x = 0; x < abs(m); x++) {
        sum += n;
    }
    return sum;
}

int main() {
    int a, b, result, check;
    printf("Enter the two numbers to be multiplied: ");
    std::cin >> a >> b;
    

    result = multiplacation(a, b);
    check = a * b;

    if (result == check) {
        printf("The result checks out| check: %d \t result: %d\n\n", check, result);
    }

    return 0;
}