### regresja logistyczna
#   regresja logistyczna    najdaje się tylko wtedy gdy zmienna opisowa ma 2 stany np 1 i 0
#                     1
#       y   =     --------------
#           1+ e -(a + (b0*x0 +b1*x1+ b2*x2) )
#                   p
#   Odds(p ) =    ------
#                 1 - p
#                   /   p    \
#   logit(p) =  ln |  ------  |
#                   \ 1 - p  /
#              1
#    p  =  ---------
#                 -logit(p)
#           1 + e
#
#
#                  /     p     \
#  logit(p) = ln  |    ------   |   = b0 + b1x1 + b2x2 + b3x3+ ....bnxn
#                  \   1 - p   /
#                           1
#           p   =       ----------
#                              -(b0 + b1x1 + b2x2 + b3x3+ ....bnxn)
#                         1 + e
#  python! 3.10
from numpy import arange
from scipy.optimize import curve_fit
from pylab import plot,show
def delogit(x,b0,b1):
    e = 2.7182
    return 1.0 / (1.0 + e**(-(b0 + b1*x)))
x = arange(0,20)
y = [0,0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1]
b,pcov = curve_fit(delogit,x,y)
line = delogit(x,b[0],b[1])
print(b)
plot(x,line,"r-",x,y,"o")
show()