from numpy import *
#A few classes of various integration methods used in computational physics/math
class Trapazoid:

    def __init__(self,f, A, B, N):
        self.f=f
        self.A=A
        self.B=B
        self.N=N
    def get_trap(self):
        h=(self.B-self.A)/(self.N-1)
        sum=(self.f(self.A)+self.f(self.B))/2

        for i in range(1, self.N-1):
            sum+=self.f(self.A+i*h)
        sum=h*sum
        return sum
class Simpson(Trapazoid):

    def __init__(self, f, A, B, N):
        Trapazoid.__init__(self, f, A, B, N)
    def get_simpson(self):
        h=(self.B-self.A)/self.N
        sum=self.f(self.A)+self.f(self.B)

        for i in range(1, self.N):
            k=self.A+i*h
            if i%2==0:
                sum=sum+2*self.f(k)
            else:
                sum=sum+4*self.f(k)
        sum=sum*h/3
        return sum
class Gaussian(Trapazoid):

    def __init__(self, f, A, B, N, x, w):
        Trapazoid.__init__(self, f, A, B, N)
        self.x=x
        self.w=w
    def gauss(self):
        job = int(input("Enter the job number: "))
        m = i = j = t = t1 = pp = p1 = p2 = p3 = 0.0
        eps= finfo(float).eps
        m=int((self.N+1)/2)
        for i in range(1, m+1):
            t=cos(pi*(float(i)-0.25)/(float(self.N)+0.5))
            t1=1
            while (abs(t-t1)>=eps):
                p1=1;p2=0
                for j in range(1, self.N+1):
                    p3=p2;p2=p1
                    p1=((2.0*float(j)-1)*t*p2-(float(j)-1.0)*p3)/(float(j))
                pp = self.N*(t*p1-p2)/(t*t-1)
                t1=t; t=t1-p1/pp
            self.x[i-1]=-t;self.x[self.N-i]=t
            self.w[i-1]=2.0/((1.0-t*t)*pp*pp)
            self.w[self.N-i]=self.w[i-1]
        if job==0:  #a+b/2=midpoint [a,b]
            for i in range(0, self.N):
                self.x[i]=self.x[i]*(self.B-self.A)/2.0+(self.B+self.A)/2.0
                self.w[i]=self.w[i]*(self.B-self.A)/2.0
        if job==1:  #[0 ->inf] a=midpoint
            for i in range(0, self.N):
                xi=self.x[i]
                self.x[i]=self.A*(1+xi)/(1-xi)
                self.w[i]=self.w[i]*(2.0*self.A)/((1-xi)*(1-xi))
        if job==2:  #[-inf -> inf] scale set by a
            for i in range(0, self.N):
                xi=self.x[i]
                self.x[i]=self.A*xi/(1-xi*xi)
                self.w[i]=self.w[i]*(self.A*(1+xi*xi))/((1-xi*xi)*(1-xi*xi))
        if job==3:  #[a->inf] a+2b=midpoint
            for i in range(0, self.N):
                xi=self.x[i]
                self.x[i]=(self.A+2.0*self.B+self.A*xi)/(1-xi)
                self.w[i]=self.w[i]*(2.0*(self.B+self.A))/((1-xi)*(1-xi))
        if job==4:  #[0->b] ab/(b+a)=midpoint
            for i in range(0, self.N):
                xi=self.x[i]
                self.x[i]=(self.A*self.B*(1+xi))/(self.B+self.A-(self.B-self.A)*xi)
                self.w[i]=self.w[i]*(2.0*self.A*self.B*self.B)/((self.B+self.A-(self.B-self.A)*xi)*(self.B+self.A-(self.B-self.A)*xi))
        quadra = 0

        for n in range(0, self.N):
            quadra+=self.f(self.x[n])*self.w[n]
        return quadra


class mean_value(Trapazoid):
    def __init__(self, f, A,B,N, x):
        Trapazoid.__init__(self, f, A, B, N)
        self.x=x
    def mean_value(self):
        h = (self.B-self.A)/self.N
        sum=0
        for i in range(0, self.N):
            meanv=self.f(self.x[i]+i*h)
            sum+=meanv
        return sum*h