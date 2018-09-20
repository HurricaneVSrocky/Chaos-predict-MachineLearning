import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
import segypy
import numpy as np
import copy
from sklearn.svm import SVR  # 调用支持向量机的分类回归函数NuSVR(支持向量回归)

def read_segy_file(FileName):
    filename=FileName
    # Set verbose level
    segypy.verbose=1
    SH = segypy.getSegyHeader(filename)
    #%% Read Segy File
    [Data,SH,STH]=segypy.readSegy(filename)
    print('数组大小：',np.shape(Data))
    return Data,SH,STH

'''提取每条缆的数据'''
def pick_line(Data,SH):
    RowNum, ColumNum = np.shape(Data)  # 获取模拟地震记录的行列数
    print('数据的行数：', RowNum)
    print('数据的列数：', ColumNum)
    Lines=np.hsplit(Data,10)
    print('一条线的数组大小：',np.shape(Lines[1]))
    NewSH=copy.deepcopy(SH)
    NewSH['DataTracePerEnsemble']=ColumNum//10
    NewSH['ntraces']=ColumNum//10
    print(SH)
    print(NewSH)
    return Lines,NewSH

''' 数据预处理——归一化 '''
def Normalization(Chaos):
    Xmax = np.max(Chaos)
    Xmin = np.min(Chaos)
    ChaosNorm = 2 * (Chaos - Xmin) / (Xmax - Xmin) - 1
    return ChaosNorm

'''相空间重构'''
def PhaSpaRecon(Chaos, tau, m):
    # 输入：Chaos=混度序列；tau=重构时延；m=嵌入维度
    # 输出：xn=相空间中的点序列（每一列为相空间中的一个点）；dn=一步预测的目标

    Length = len(Chaos)  # 将序列长度存储下来
    print('混沌序列的长度：', Length)
    ChaosNew = np.reshape(Chaos, (1, Length))
    print('改变Chaos的shape：', np.shape(ChaosNew))
    if Length - 1 - (m - 1) * tau < 1:
        print('delay time or the embedding dimension is too large')
        xn = []
        dn = []
    else:
        xn = np.empty((m, Length - 1 - (m - 1) * tau))
        print('预留的相空间的大小：', np.shape(xn))
        for i in range(m):
            xn[i, :] = ChaosNew[0, (i + 1 - 1) * tau:Length - 1 - (m - (i + 1)) * tau]
        print('重构的序列：', np.transpose(xn))
        print('相空间的点序列大小：', np.shape(xn))

        dn = ChaosNew[0, 1 + (m - 1) * tau:Length]
        d_len = len(dn)
        print('一步预测长度：',d_len)
        # dnReshape=np.reshape(dn,(1,d_len))
    return xn, dn

'''test'''
filename='pengyin.sgy'
[Data,SH,STH]=segypy.readSegy(filename)
[Lines,NewSH]=pick_line(Data,SH)
#%% Plot Segy file
scale=1e-9

# wiggle plot
#segypy.wiggle(Lines[0],NewSH,4)

TestLine=Lines[0]
TestTrace=TestLine[:300,401] #第400道的前300个点
print('测试道大小：',np.shape(TestTrace))
Num_Train = 100  # 训练样本点数
Num_Test = 100  # 测试样本点数

ChaosTrain = TestTrace[:Num_Train]
ChaosTrain = ChaosTrain[:, np.newaxis]  # 训练样本
print(np.shape(ChaosTrain))

ChaosTest = TestTrace[Num_Train:Num_Train + Num_Test]
ChaosTest = ChaosTest[:, np.newaxis]  # 测试样本
print(np.shape(ChaosTest))

d=3
t=1

Xn_Tr, Dn_Tr = PhaSpaRecon(ChaosTrain, t, d)
Xn_Te, Dn_Te = PhaSpaRecon(ChaosTest, t, d)


''' Fit regression model '''
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)  # 核函数采用高斯径向基核函数（Radial Basis Function-RBF）

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