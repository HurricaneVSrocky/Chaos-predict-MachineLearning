import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.matlib import repmat
from scipy.integrate import odeint
import pypsr
from sklearn.svm import SVR  # 调用支持向量机的分类回归函数NuSVR(支持向量回归)


t=np.arange(0,2000,1)
y=np.sin(t)

Chaos = y  # 混沌序列
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



'''相空间重构'''
def PhaSpaRecon_multi(Chaos, tau, m):
    # 输入：Chaos=混度序列；tau=重构时延；m=嵌入维度
    # 输出：xn=相空间中的点序列（每一列为相空间中的一个点）；dn=一步预测的目标

    Length = len(Chaos)  # 将序列长度存储下来
    print('混沌序列的长度：', Length)
    ChaosNew = np.reshape(Chaos, (1, Length))
    print('改变Chaos的shape：', np.shape(ChaosNew))
    print('改变shape之后的Chaos：',ChaosNew)

    if Length  - (m - 1) * tau < 1:
        print('delay time or the embedding dimension is too large')
        xn = []

    else:
        xn = np.empty((m, Length -(m - 1) * tau))
        print('预留的相空间的大小：', np.shape(xn))
        for i in range(m):
            xn[i, :] = ChaosNew[0, 1+(i+1) * tau-tau-1:Length -m*tau+(i+1)*tau]
        print('重构的序列：', np.transpose(xn))
        print('相空间的点序列大小：', np.shape(xn))
    return xn

def PhaSpaRecon_one(Chaos, tau, m):
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

        print('The shape of dn:',np.shape(dn))
        # dnReshape=np.reshape(dn,(1,d_len))
    return xn, dn


'''test'''
Num_Train =1000  # 训练样本点数
Num_Test = 100  # 测试样本点数

ChaosTrain = Chaos[:Num_Train, 0]
ChaosTrain = ChaosTrain[:, np.newaxis]  # 训练样本
print(np.shape(ChaosTrain))
print('混沌序列：',np.transpose(ChaosTrain))

ChaosTest = Chaos[Num_Train:Num_Train + Num_Test, 0]
ChaosTest = ChaosTest[:, np.newaxis]  # 测试样本
print(np.shape(ChaosTest))

'''各种参数'''
d=8 #嵌入维数
t=8 #延迟时间

Xn_Tr, Dn_Tr = PhaSpaRecon_one(ChaosTrain, t, d) #对训练样本进行相空间重构



''' Fit regression model '''
svr_rbf = SVR(C=1e3, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)  # 核函数采用高斯径向基核函数（Radial Basis Function-RBF）

Num_Pr=100 #对未来100个点进行预测

X_st=Chaos[Num_Train-(d-1)*t-1:Num_Train]
print('混沌序列：',X_st)

PredictResults_Pr=np.empty(Num_Pr)


'''循环多步预测'''
for i in np.arange(Num_Pr):
    print('第',i+1,'步预测')
    XN_st=PhaSpaRecon_multi(X_st,t,d)
    PredictResults_Pr[i] = svr_rbf.fit(np.transpose(Xn_Tr), Dn_Tr).predict(np.transpose(XN_st)) #以训练样本的重构序列进行训练产生模型，进行预测
    print('单次测的值：',PredictResults_Pr[i],type(PredictResults_Pr[i]))
    List_X_st=X_st[1:,0].tolist() #将预测出的值添加到混沌序列的末尾，参加下次重构
    List_X_st.append(PredictResults_Pr[i])
    X_st=np.array(List_X_st)
    X_st=np.reshape(X_st,(len(X_st),1))
    print(X_st)

PredictTarget=Chaos[Num_Train:Num_Train+Num_Pr] #预测目标

'''预测误差'''
Error_pr=np.transpose(PredictTarget)-PredictResults_Pr
'''预测结果作图'''
plt.figure(4)
plt.plot(np.transpose(PredictResults_Pr), 'b', label='Pre_Pr(t)')
plt.legend(loc='best')
plt.title('Predict results')
plt.xlabel('t')
plt.grid()

'''训练样本预测结果与原始序列对比'''
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(5)
plt.subplot(2, 1, 1)  # 预测结果与真实值曲线对比
plt.plot(PredictTarget, '-b', label='Original')
plt.plot(np.transpose(PredictResults_Pr), '--r', label='Predict')
plt.legend(loc='lower right')
plt.xlabel('t')
plt.title('训练样本真实值（.）与预测值(-)')

plt.subplot(2, 1, 2)  # 预测误差曲线
plt.plot(np.transpose(Error_pr), 'b', label='Error')
plt.legend(loc='best')
plt.title('预测误差曲线')
plt.xlabel('t')
plt.grid()
plt.show()