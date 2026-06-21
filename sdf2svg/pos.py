import math
import numpy as np


def sqr(x):
    return x * x


class pos(object):
    def __init__(self, x=None, y=None):
        if(x is None)and(y is None):
            self.x=0
            self.y=0
        elif(y is None):
            if isinstance(x,pos):
                self.x=x.x
                self.y=x.y
            else:
                self.x=x[0]
                self.y=x[1]
        else:
            self.x = x
            self.y = y

    def __add__(self, rval):
        x = self.x + rval.x
        y = self.y + rval.y
        return pos(x, y)

    def __sub__(self, rval):
        x = self.x - rval.x
        y = self.y - rval.y
        return pos(x, y)

    def __rmul__(self, other):
        x = other * self.x
        y = other * self.y
        return pos(x, y)

    def __mul__(self, other):
        x = self.x * other
        y = self.y * other
        return pos(x, y)

    def __abs__(self):
        return math.sqrt(sqr(self.x) + sqr(self.y))

    def __len__(self):
        return self.__abs__()

    def norm(self):
        if(abs(self.x)<1e-8)and(abs(self.y)<1e-8):
            return pos(0,0)
        else:
            l=abs(self)
            return (1.0/l)*self

    def print(self):
        print('(',self.x,',',self.y,')',end='')

    def __repr__(self):
        return '('+str(self.x)+','+str(self.y)+')'

    def tonparray(self):
        return np.array((self.x, self.y))
