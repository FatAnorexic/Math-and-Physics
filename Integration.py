
class Integration:
    def __init__(self, f, a, b, n):
        self.f=f; self.a=a; self.b=b; self.n=n
    def trapazoid(self):
        h=(self.b-self.a)/(self.n-1)
        sum=(self.f(self.a)+self.f(self.b))/2

        for i in range(1, self.n-1):
            sum+=self.f(self.a+i*h)
        sum=h*sum
        return sum
    def simpson(self):
        h=(self.b-self.a)/self.n
        sum=self.f(self.a)+self.f(self.b)

        for i in range(1, self.n):
            k=self.a+i*h
            if i%2==0:
                sum+=2*self.f(k)
            else:
                sum+=4*self.f(k)
        sum=sum*h/3
        return sum
