import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.matlib import repmat
from scipy.integrate import odeint
import pypsr
from sklearn.svm import SVR  # 调用支持向量机的分类回归函数NuSVR(支持向量回归)


'''求解混沌方程的微分方程'''
# x''+a*x'-x+x^3=f*cos(t)--Duffing混沌序列

def Duffing(InitialValue, t, a, f):
    x, y = InitialValue
    dydt = [y, x - np.power(x, 3) - a * y + f * np.cos(t)]
    return dydt


y0 = [0, 0]  # 初值条件

t = np.linspace(0, 299, 300)
print(t)

a = 0.25  # 阻尼系数
f = 0.45  # 激励系数

Sol = odeint(Duffing, y0, t, args=(a, f))  # 解微分方程

Chaos = Sol[:, 0]  # 混沌序列
Chaos = Chaos[:, np.newaxis]

print('混沌序列的大小：', np.shape(Chaos))
print('混沌序列数据类型：', type(Chaos))

'''混沌序列作图'''
plt.figure(1)
plt.plot(t, Chaos, 'b', label='x(t)')
plt.title('Chaos')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()

Num_Train = 100  # 训练样本点数
Num_Test = 100  # 测试样本点数

ChaosTrain = Chaos[:Num_Train, 0]
ChaosTrain = ChaosTrain[:, np.newaxis]  # 训练样本
print(np.shape(ChaosTrain))
print('混沌序列：',np.transpose(ChaosTrain))

ChaosTest = Chaos[Num_Train:Num_Train + Num_Test, 0]
ChaosTest = ChaosTest[:, np.newaxis]  # 测试样本
print(np.shape(ChaosTest))


''' 数据预处理——归一化 '''
def Normalization(Chaos):
    Xmax = np.max(Chaos)
    Xmin = np.min(Chaos)
    ChaosNorm = 2 * (Chaos - Xmin) / (Xmax - Xmin) - 1
    return ChaosNorm


def Normalization1(Chaos):  # 均值0 方差1
    rows, cols = np.shape(Chaos)
    if rows == 1:
        Chaos = np.transpose(Chaos)
        len = cols
        num = 1
    else:
        len = rows
        num = cols
    MeanChaos = np.mean(Chaos)
    Chaos = Chaos - repmat(MeanChaos, len, 1)  # 0均值
    w = 1 / np.std(Chaos, ddof=1)
    ChaosOutput = np.multiply(Chaos, repmat(w, len, 1))  # 方差1
    return ChaosOutput


def Z_ScoreNormalization(Chaos):
    mu = np.average(Chaos)
    Sigma = np.std(Chaos)
    NormChaos = (Chaos - mu) / Sigma
    return NormChaos


ChaosTrainNorm = Normalization(ChaosTrain)
ChaosTestNorm = Normalization(ChaosTest)

'''归一化混沌序列作图'''
plt.figure(3)
plt.plot(ChaosTrainNorm, 'b', label='x(t)')
plt.title('ChaosTrNorm')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()

plt.figure(7)
plt.plot(ChaosTestNorm, 'b', label='x(t)')
plt.title('ChaosTeNorm')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()

''' 相空间重构（坐标延迟法 Takens定理） '''

def PhaSpaRecon(Chaos, tau, m):
    # 输入：Chaos=混度序列；tau=重构时延；m=嵌入维度
    # 输出：xn=相空间中的点序列（每一列为相空间中的一个点）；dn=一步预测的目标

    Length = len(Chaos)  # 将序列长度存储下来
    print('Input Chaos:',Chaos)
    print('混沌序列的长度：', Length)
    ChaosNew = np.reshape(Chaos, (1, Length))
    print('改变Chaos的shape：', np.shape(ChaosNew))
    print('ChaosNew:',ChaosNew)
    if Length  -1- (m - 1) * tau < 1:
        print('delay time or the embedding dimension is too large')
        xn = []
        dn = []
    else:
        xn = np.empty((m, Length  -1- (m - 1) * tau))
        print('预留的相空间的大小：', np.shape(xn))
        for i in range(m):
            xn[i, :] = ChaosNew[0, (i + 1 - 1) * tau:Length - 1 - (m - (i + 1)) * tau]
        print('重构的序列：', np.transpose(xn))
        print('相空间的点序列大小：', np.shape(xn))

        dn = ChaosNew[0, 1 + (m - 1) * tau:Length]
        d_len = len(dn)
        print('一步预测的目标',dn)
        # dnReshape=np.reshape(dn,(1,d_len))
    return xn, dn

d=3
t=1

Xn_Tr, Dn_Tr = PhaSpaRecon(ChaosTrainNorm, t, d)
Xn_Te, Dn_Te = PhaSpaRecon(ChaosTestNorm, t, d)

print('重构的序列：',np.transpose(Xn_Tr))

print('一步预测目标的大小：', np.shape(Dn_Tr))

'''一步预测目标作图'''
plt.figure(2)
plt.plot(np.transpose(Dn_Tr), 'b', label='Dn_Pr(t)')
plt.legend(loc='best')
plt.title('one step predict target')
plt.xlabel('t')
plt.grid()

''' Fit regression model '''
svr_rbf = SVR(C=1e4, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)  # 核函数采用高斯径向基核函数（Radial Basis Function-RBF）

PredictResults_Tr = svr_rbf.fit(np.transpose(Xn_Tr), Dn_Tr).predict(np.transpose(Xn_Tr))
PredictResults_Te = svr_rbf.fit(np.transpose(Xn_Tr), Dn_Tr).predict(np.transpose(Xn_Te))

'''预测误差'''
Error_Tr = Dn_Tr - PredictResults_Tr
Error_Te = Dn_Te - PredictResults_Te

'''预测结果作图'''
plt.figure(4)
plt.plot(np.transpose(PredictResults_Tr), 'b', label='Pre_Pr(t)')
plt.legend(loc='best')
plt.title('Predict results')
plt.xlabel('t')
plt.grid()

'''训练样本预测结果与原始序列对比'''
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(5)
plt.subplot(2, 1, 1)  # 预测结果与真实值曲线对比
plt.plot(np.transpose(Dn_Tr), '-b', label='Original')
plt.plot(np.transpose(PredictResults_Tr), '--r', label='Predict')
plt.legend(loc='lower right')
plt.xlabel('t')
plt.title('训练样本真实值（.）与预测值(-)')

plt.subplot(2, 1, 2)  # 预测误差曲线
plt.plot(Error_Tr, 'b', label='Error')
plt.legend(loc='best')
plt.title('预测误差曲线')
plt.xlabel('t')
plt.grid()


'''测试样本预测结果与原始序列对比'''
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(6)
plt.subplot(2, 1, 1)  # 预测结果与真实值曲线对比
plt.plot(np.transpose(Dn_Te), '-b', label='Original')
plt.plot(np.transpose(PredictResults_Te), '--r', label='Predict')
plt.legend(loc='lower right')
plt.xlabel('t')
plt.title('测试样本真实值（.）与预测值(-)')

plt.subplot(2, 1, 2)  # 预测误差曲线
plt.plot(Error_Te, 'b', label='Error')
plt.legend(loc='best')
plt.title('预测误差曲线')
plt.xlabel('t')
plt.grid()
plt.show()
