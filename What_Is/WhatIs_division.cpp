/*
If you've seen my file on what multiplacation really is at its core, you may be venture to
make the next guess at what division is fundamentally. That is, if multiplacation is simply
advanced addition, then what is division? You'd be correct in asserting that it's simply
advanced subtraction. In fact any mathmatical operation, with enough boiler plate, can devolve
into either addition or subtraction. Ultimately what our computers do when calculating for us.

However, this advanced subtraction comes with a few caviates| to start lets look at what it is:
20/4 is simply asking the question, how many times would you have to subtract 4 from 20 in order
reach zero->20-4=16|1->16-4=12|2->12-4=8|3->8-4=4|4->4-4=0|5==> 5 times 20/4=5. But what is 4/20?
In other words: How many times must you subtract 20 from 4 to get to zero? The answer isn't as straight
forward as multiplacation where the order of terms didn't matter; here the order makes all the differece.

To answer 4/20, we need to ask-> what fraction of 20 could we subtract from 4 to equal 20? This leads us
into fractions which will be covered in a later what is, but the answer to satisfy curriosity is 20% of 20
or 4.

similarly, 0/20 is simply 0-0 twenty times==0. however, 20/0? Well you'd need to subtract 0 a seemingly
infinite number of times to get 20 to 0, but technically this isn't entirely accurate as it depends on the
sign and direction on a number graph you approach from. We say it is undefined, and has a better explanation
with limits, which we will get to at a much later what is.

This may seem like a lot, but I feel it's extremely important in building on mathematical foundations,
especially in children. So with all that said, let us make a simple subtraction loop. One that reduces any
number to zero and spits out a remainder. It will throw errors for fractional numbers IE | 4/20 and of
course for numbers divided by 0.
*/

#include <iostream>
#include <cstdlib>

//Right off the bat, we may find ourselves with a remainder: IE 22/4=>5 with a remainder of 2
//Given we cannot return multiple values in c++ we must build a structure.
struct divisors {
    int result = 0, remainder = 0;
};
//And we must define that structure as a returnable data type.
typedef struct divisors Struct;

Struct division(int a, int b) {
    //Simple checks|if its 0 we exit and if we have float results we exit
    if (b == 0) {
        printf("You cannot divide by zero: in this program it is undefined\n");
        exit(-1);
    }
    if (b > a) {
        printf("You cannot do fractional division with this program. This may be included later\n");
        exit(-1);
    }

    Struct d;
    while (a > 0) {
        if (a - b < 0) {
            //essentially what a remainder is->the absolute value of the total amount left over after
            //going below zero. 
            d.remainder += abs(a - b);
            return d;
        }
        a -= b;
        d.result++;
        
    }
    return d;

}

int main() {
    int a, b;
    Struct result;
    //As part of my continuing development, I'm trying f strings to see if it's any cleaner
    printf("Enter two numbers| one to be divided the other as the divisor: \n");
    scanf_s("%d%d", &a, &b);

    result = division(abs(a), abs(b));

    //Initially I intended not to allow negative numbers, but when I realized it'd be a simple check and 
    //-1, I decided it'd be worth the time and effort to include
    if (a < 0 || b < 0) {
        result.result *= -1;
    }

    printf("The results are %d with a remainder of %d", result.result, result.remainder);

    return 0;

}