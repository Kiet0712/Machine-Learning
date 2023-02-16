import math
import numpy as np

#tìm cực trị của hàm số f(x)=x^3+sinx
def gd1(x):
    return 3*x**2-math.cos(x)
def solvegd1(eta, x_init):
    x = [x_init]
    for i in range(500):
        x_new = x[-1]-eta*gd1(x[-1])
        if abs(gd1(x_new)) < 1e-6:
            break
        x.append(x_new)
    return x[-1]
#sử dụng momentum cho hàm có nhiều cực trị
#xét hàm x^2+10sinx
def gd2(x):
    return 2*x+10*math.cos(x)
def solvegd2(eta, gamma, x_init):
    x = [x_init]
    v = [0]
    for i in range(500):
        v_new = v[-1]*gamma+eta*gd2(x[-1])
        x_new = x[-1]-v_new
        if abs(gd2(x_new)) < 1e-6:
            break
        x.append(x_new)
        v.append(v_new)
    return x[-1]
print(solvegd2(0.1,0.9,-2))