import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
import segypy
import numpy as np
import copy
from sklearn.neural_network import MLPRegressor


'''提取每条缆的数据'''
def pick_line(Data,SH,perTrace):
    #perTrace=每条缆的道数
    RowNum, ColumNum = np.shape(Data)  # 获取模拟地震记录的行列数
    print('数据的行数：', RowNum)
    print('数据的列数：', ColumNum)
    Num_Trace=ColumNum//perTrace
    Lines=np.hsplit(Data,Num_Trace)  #按缆将数据矩阵分开
    print('一条线的数组大小：',np.shape(Lines[1]))
    NewSH=copy.deepcopy(SH)
    NewSH['DataTracePerEnsemble']=ColumNum//Num_Trace
    NewSH['ntraces']=ColumNum//Num_Trace
    print(SH)
    print(NewSH)
    return Lines,NewSH

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

def Normalization(Chaos):
    Xmax = np.max(Chaos)
    Xmin = np.min(Chaos)
    ChaosNorm = 2 * (Chaos - Xmin) / (Xmax - Xmin) - 1
    return ChaosNorm

'''test'''
filename='pengyin.sgy'
[Data,SH,STH]=segypy.readSegy(filename)
perTrace=480  #每条缆的道数
[Lines,NewSH]=pick_line(Data,SH,perTrace)
#%% Plot Segy file
scale=1e-9

# wiggle plot
#segypy.wiggle(Lines[0],NewSH,4)

TestLine=Lines[0] #选取测试线
TestTrace=TestLine[:,401] #第400道作为测试道
TestTraceNorm=Normalization(TestTrace)
print('测试道大小：',np.shape(TestTrace))
Num_Train =1000  # 训练样本点数
Num_Test = 1000  # 测试样本点数

ChaosTrain = TestTrace[:Num_Train]
ChaosTrain = ChaosTrain[:, np.newaxis]  # 训练样本
print(np.shape(ChaosTrain))

ChaosTest = TestTrace[Num_Train:Num_Train + Num_Test]
ChaosTest = ChaosTest[:, np.newaxis]  # 测试样本
print(np.shape(ChaosTest))

'''各种参数'''
d=10 #嵌入维数
t=10 #延迟时间

Xn_Tr, Dn_Tr = PhaSpaRecon_one(ChaosTrain, t, d) #对训练样本进行相空间重构





Num_Pr=50 #对未来50个点进行预测

X_st=TestTrace[Num_Train-(d-1)*t-1:Num_Train,np.newaxis]
print('混沌序列：',X_st)

PredictResults_Pr=np.empty(Num_Pr)

''' Fit regression model '''
NN_Regressor = MLPRegressor(activation = 'relu',solver = 'adam',alpha = 0.0001,batch_size = 'auto',learning_rate = 'constant',
                            learning_rate_init = 0.001,power_t = 0.5,max_iter = 200,shuffle = True,random_state = None,tol = 0.0001,verbose = False,warm_start = False,momentum = 0.9,
                            nesterovs_momentum = True,early_stopping = False,validation_fraction = 0.1,beta_1 = 0.9,beta_2 = 0.999,epsilon = 1e-08)

'''循环多步预测'''
for i in np.arange(Num_Pr):
    print('第',i+1,'步预测')
    XN_st=PhaSpaRecon_multi(X_st,t,d)
    PredictResults_Pr[i] = NN_Regressor.fit(np.transpose(Xn_Tr), Dn_Tr).predict(np.transpose(XN_st)) #以训练样本的重构序列进行训练产生模型，进行预测
    print('单次测的值：',PredictResults_Pr[i],type(PredictResults_Pr[i]))
    List_X_st=X_st[1:,0].tolist() #将预测出的值添加到混沌序列的末尾，参加下次重构
    List_X_st.append(PredictResults_Pr[i])
    X_st=np.array(List_X_st)
    X_st=np.reshape(X_st,(len(X_st),1))
    print(X_st)

PredictTarget=TestTrace[Num_Train:Num_Train+Num_Pr] #预测目标

'''预测误差'''
Error_pr=PredictTarget-PredictResults_Pr

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
plt.plot(Error_pr, 'b', label='Error')
plt.legend(loc='best')
plt.title('预测误差曲线')
plt.xlabel('t')
plt.grid()
plt.show()