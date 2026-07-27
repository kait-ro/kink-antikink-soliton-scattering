from parameters import Params
import numpy as np 

class Field:
    def V(phi, p: Params):
        return p.lam/4 * (phi**2 - p.v**2)**2

    def Vprime(phi, p: Params):
        return p.lam*phi*(phi**2-p.v**2)

    def staticKink(x, x_0, p: Params, sign=1):
        if sign==-1:
            return p.v*np.tanh((p.lam/2)**(1/2)*p.v*(x-x_0))*(-1)
        else:
            return p.v*np.tanh((p.lam/2)**(1/2)*p.v*(x-x_0))

