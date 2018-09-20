import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
import segypy
import numpy as np
import copy
filename='pengyin.sgy'
# Set verbose level
segypy.verbose=1
SH = segypy.getSegyHeader(filename)
#%% Read Segy File
[Data,SH,STH]=segypy.readSegy(filename)

print(np.shape(Data))
RowNum,ColumNum=np.shape(Data)   #获取模拟地震记录的行列数
print('数据的行数：',RowNum)
print('数据的列数：',ColumNum)

'''提取某一条缆的数据'''
FirstLineData=Data[:,:ColumNum//10]
ThirdLineData=Data[:,2*ColumNum//10:3*ColumNum//10+1]
Lines=np.hsplit(Data,10)
print(Lines)
print(np.shape(Lines))
print(np.shape(Lines[1]))
print(np.shape(FirstLineData))
FSSH=copy.deepcopy(SH)
FSSH['DataTracePerEnsemble']=ColumNum//10
FSSH['ntraces']=ColumNum//10
print(SH)
print(FSSH)

#%% Plot Segy file
scale=1e-9

# wiggle plot
segypy.wiggle(FirstLineData,FSSH,4)
segypy.wiggle(ThirdLineData,FSSH,4)
#plt.figure(2)
#segypy.wiggle(DataWithNoise,SH,1)  #加噪地震剖面

#plt.figure(3)
#segypy.wiggle(GaussianNoise,SH,1) #背景噪声

# image plot
#segypy.image(FirstSourceData,FSSH,2)

#plt.figure(5)
#segypy.image(DataWithNoise,SH,3)

#segypy.wiggle(Data,SH,4)
#segypy.image(Data,SH,2)