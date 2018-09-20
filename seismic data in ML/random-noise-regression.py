import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
import segypy
import numpy as np
from scipy.integrate import odeint
import pypsr
from sklearn.svm import SVR  #调用支持向量机的分类回归函数SVR(支持向量回归)
filename='Model+GathEP-Z.sgy'
# Set verbose level
segypy.verbose=1
SH = segypy.getSegyHeader(filename)
#%% Read Segy File
[Data,SH,STH]=segypy.readSegy(filename)
print(SH)
print(np.shape(Data))
RowNum,ColumNum=np.shape(Data)   #获取模拟地震记录的行列数
print('数据的行数：',RowNum)
print('数据的列数：',ColumNum)

#求解混沌方程的微分方程
#x''+a*x'-x+x^3=f*cos(t)--Duffing混沌序列
def Duffing(InitialValue,t,a,f):
    x,y=InitialValue
    dydt=[y,x-np.power(x,3)-a*y+f*np.cos(t)]
    return dydt
y0=[np.pi-0.1,0]
t=np.linspace(0,1000,1000)
a=0.25
f=0.45
Sol=odeint(Duffing,y0,t,args=(a,f))
print(np.shape(Sol))
plt.plot(t, Sol[:, 0], 'b', label='x(t)')
#plt.plot(t, Sol[:, 1], 'g', label='y(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()

#Generator Random Noise
GaussianNoise=np.random.normal(0,0.3,(RowNum,ColumNum))  #高斯白噪声(均值0，方差0.3)

# Generate sample data
FirstColumNoise=GaussianNoise[:,0]   #第一道白噪声
plt.figure(6)
plt.plot(FirstColumNoise)  #显示第一道高斯白噪声
print('第一道噪声shape:',np.shape(FirstColumNoise))

NumberOfDots=int(np.shape(FirstColumNoise)[0]) #获得第一道噪音的长度（点数）
print('第一道噪声的长度（点数）：',NumberOfDots)
print('样本集点数：',int(np.floor(NumberOfDots*0.3)))

#设置样本的长度
Sample_Rate=0.5
Sample_Length=int(np.floor(NumberOfDots*Sample_Rate))

#取第一道噪音的前百分之Sample_Rate的点作为样本
Sample_X=np.arange(0,Sample_Length) .reshape(Sample_Length,1)
print('样本大小：',np.shape(Sample_X))
Sample_Y=FirstColumNoise[0:Sample_Length]
print(np.shape(Sample_Y))
plt.figure(9)
plt.plot(Sample_Y) #画出样本
plt.title('训练样本')

Full_Time=np.argsort(FirstColumNoise).reshape(NumberOfDots,1)
print('输入向量大小：',np.shape(Full_Time))

'''Fit regression model'''
svr_rbf=SVR(kernel='rbf',C=1e3,gamma=0.1) #核函数采用高斯径向基核函数（Radial Basis Function-RBF）

PredictResults_rbf=svr_rbf.fit(Sample_X,Sample_Y).predict(Full_Time)

plt.figure(7)
plt.plot(PredictResults_rbf)

plt.figure(8)
plt.plot(np.arange(0,NumberOfDots),np.ravel(PredictResults_rbf),color='r',label='RBF predict')
plt.plot(np.arange(0,NumberOfDots),FirstColumNoise,label='original Noise')
plt.title('Support Vector Regression')
plt.legend()
plt.show()


DataWithNoise=Data+GaussianNoise  #数据加噪






#%% Plot Segy filwe
scale=1e-9

# wiggle plot
plt.figure(1)
segypy.wiggle(Data,SH,1)  #原始地震剖面

plt.figure(2)
segypy.wiggle(DataWithNoise,SH,1)  #加噪地震剖面

plt.figure(3)
segypy.wiggle(GaussianNoise,SH,1) #背景噪声

# image plot
plt.figure(4)
segypy.image(Data,SH,2)

plt.figure(5)
segypy.image(DataWithNoise,SH,3)



























segypy.writeSegyStructure(file_name,new_data,SH,STH) #写入新segy文件