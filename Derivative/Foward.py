

class Forward:

    def __init__(self,y, t, h):
       self.y=y
       self.t=t
       self.h=h

    def set_y(self, y):
        self.y=y
    def set_t(self, t):
        self.t=t
    def set_h(self, h):
        self.h=h

    def get_y(self):
        return self.y
    def get_t(self):
        return self.t
    def get_h(self):
        return self.h
    def get_forward_diff(self):
        return (self.y(self.t+self.h)-self.y(self.t))/self.h
    def get_central(self):
        return (self.y(self.t + self.h / 2) - self.y(self.t - self.h / 2)) / self.h
    def get_extrapolated(self):
        return (8*(self.y(self.t+self.h/4)-self.y(self.t-self.h/4))-(self.y(self.t+self.h/2)-self.y(self.t-self.h/2)))/(3/self.h)

class Second_Derriv(Forward):
    def __init__(self, y, t, h):
        Forward.__init__(self, y, t, h)
    def get_second_deriv(self):
        return (self.y(self.t+self.h)+self.y(self.t-self.h)-2*self.y(self.t))/(self.h**2)

    def second_deriv_expanded(self):
        return ((self.y(self.t+self.h)-self.y(self.t))-(self.y(self.t)-self.y(self.t-self.h)))/(self.h**2)
