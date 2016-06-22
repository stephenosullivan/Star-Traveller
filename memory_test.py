import random
from math import sqrt
from collections import OrderedDict
import sys

#@profile
def my_func():
    Nstars = 500
    stars = stardict(Nstars)
    a = [OrderedDict() for _ in range(Nstars)]
    for i in stars.keys():
#	print(i)
        tmp = {j: distance(stars[i],stars[j]) for j in stars.keys() if j != i and j not in a}
        a[i] = OrderedDict(sorted(tmp.items(), key=lambda x: x[1]))
#a = {i: OrderedDict(sorted({j: distance(stars[i],stars[j]) for j in stars)}) for i in stars}
    #print([a[i] for i in range(100)])
    #b = [2] * (2*10 **7)
    #del b
    #print(a[0][1], 0 in a[0]
    print('end')
    print(sys.getsizeof(a))
    return a

def stardict(n):
    return {i: (random.choice(xrange(1000)), random.choice(xrange(1000))) for i in xrange(n)}

def distance(a, b):
    return sqrt((a[1] - b[1])**2 + (a[0] - b[0])**2)



if __name__ == '__main__':
    my_func()
