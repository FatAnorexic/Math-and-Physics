from numpy import *

w = zeros((2001), float)
x = zeros((2001), float)


class guass:


    def __init__(self,npts,a,b,x,w):
        self.npts=npts
        self.a=a
        self.b=b
        self.x=x
        self.w=w

    def guass(self, npts,job, a, b, x, w):
        self.npts=npts
        self.a=a
        self.b=b
        self.x=x
        self.w=w
        m = i = j = t = t1 = pp = p1 = p2 = p3 = 0.0
        eps=finfo(float).eps
        m=int((npts+1)/2)
        for i in range(1,m+1):
            t = cos(pi * (float(i) - 0.25) / (float(npts) + 0.5))
            t1 = 1
            while (abs(t - t1) >= eps):
                p1 = 1; p2 = 0
                for j in range(1, npts + 1):
                    p3 = p2; p2 = p1
                    p1 = ((2.0 * float(j) - 1) * t * p2 - (float(j) - 1.0) * p3) / (float(j))
                pp = npts * (t * p1 - p2) / (t * t - 1)
                t1 = t; t = t1 - p1 / pp
            x[i - 1] = -t; x[npts - i] = t
            w[i - 1] = 2.0 / ((1.0 - t * t) * pp * pp)
            w[npts - i] = w[i - 1]
        if (job == 0):
            for i in range(0, npts):
                x[i] = x[i] * (b - a) / 2.0 + (b + a) / 2.0
                w[i] = w[i] * (b - a) / 2.0
        if (job == 1):
            for i in range(0, npts):
                xi = x[i]
                x[i] = a * b * (1.0 + xi) / (b + a - (b - a) * xi)
                w[i] = w[i] * 2.0 * a * b * b / ((b + a - (b - a) * xi) * (b + a - (b - a) * xi))
        if (job == 2):
            for i in range(0, npts):
                xi = x[i]
                x[i] = (b * xi + b + a + a) / (1.0 - xi)
                w[i] = w[i] * 2.0 * (a + b) / ((1.0 - xi) * (1.0 - xi))
    def gaussint(self,no, min, max):
        self.no=no
        self.min=min
        self.max=max
        job=int(input("Enter the type of limits to implement: "))
        return guass(no,min, max,x,w)

