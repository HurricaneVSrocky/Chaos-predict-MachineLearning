#-*-coding:utf-8-*-
import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
import segypy
import numpy as np
from sympy import *

n=5000

def Logistic(x,n):
    for i in range(n):
        y = 4 * x * (1 - x)
        x = y
    return x

print()
def LE_calculate():
    a = 0.123456789  # 混沌的初始值
    count = 0
    sum_value = 0  # 初始的求和值为0
    x = symbols('x')
    expr = 4 * x * (1 - x)  # 表达式
    diff_expr = diff(expr, x)  # 对表达式进行求导,得到导数的表达式。该表达式固定（带参数）
    # 先迭代混沌方程1000次消除初始影响,以第1001次的返回值作为初值
    a = Logistic(a, 1001)
    while (count < n):
        diff_value = diff_expr.subs(x, a)  # 带入当前迭代值，得到当前的导数值(数值)
        diff_value_ln = ln(abs(diff_value))  # 对当前导数值取绝对值然后取对数
        sum_value = sum_value + diff_value_ln  # 计算求和值
        a = Logistic(a, 1)  # 每次只迭代一次，获取当前的迭代值
        count = count + 1
    LE_value = sum_value / n
    print(LE_value)

if __name__ == '__main__':
   LE_calculate()