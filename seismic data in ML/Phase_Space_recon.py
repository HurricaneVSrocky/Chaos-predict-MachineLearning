import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
from skimage import draw,data,io
import numpy as np

# y"+a*y'+b*y=0
from scipy.integrate import odeint
from pylab import *


def deriv(y, t):  # 返回值是y和y的导数组成的数组
    a = -2.0
    b = -0.1
    return array([y[1], a * y[0] + b * y[1]])


time = linspace(0.0, 50.0, 1000)
yinit = array([0.0005, 0.2])  # 初值
y = odeint(deriv, yinit, time)

figure()
plot(time, y[:, 0], label='y')  # y[:,0]即返回值的第一列，是y的值。label是为了显示legend用的。
plot(time, y[:, 1], label="y'")  # y[:,1]即返回值的第二列，是y’的值
xlabel('t')
ylabel('y')
legend()
show()

m=20
for i in range(m):
    print(i)

