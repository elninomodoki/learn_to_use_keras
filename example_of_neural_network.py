'''一个利用神经网络预报能见度的示例'''
from math import sqrt
import pandas as pd
from numpy import concatenate
import numpy as np
np.random.seed(1337)
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
import h5py
from keras.models import load_model
import sklearn
from sklearn.model_selection import train_test_split

def MaxMinNormalization(x):
	'''能见度归一化'''
	x = (x - 50.) / 90000.
	return x
	
def MaxMin_inverse(x):
	'''能见度反归一化'''
	x = 90000.*x+50.
	return x

def Normalizemaxmin(X,X_stan):
	'''（每一个数据-当前列的最小值）/(当前列的最大值-当前列的最小值)'''
	X_norm = np.zeros(X.shape)
	n = X.shape[1]
	xmax = np.zeros((1,n))
	xmin = np.zeros((1,n))
	xmax = np.max(X_stan,axis=0)
	xmin = np.min(X_stan,axis=0)
	for i in range(n):
		X_norm[:,i] = (X[:,i]-xmin[i])/(xmax[i] - xmin[i])
	return X_norm
	
	
#读取训练数据
dataset = read_csv('filename', header=None)
values = dataset.values
values1 = values.astype('float32')

#读取归一化数据
x_stan = pd.read_csv('filename', header = None , sep = ',').values
#此处使用了自己的归一化文件进行了归一化，也可使用sklearn自带的归一化方式进行归一化

#数据预处理，去掉某些列
Xx = values1[:, 3:-1]
pre = values1[:, 0]
yy = values1[:, -1]

#标签归一化
y = MaxMinNormalization(yy)

#合并季节指数和其他变量
pree = pre.reshape(pre.shape[0],1)
XX = concatenate((pree,Xx),axis = 1)

#训练变量归一化
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(XX)

#随机分为训练和测试样本
X_train, X_test, y_train, y_test = train_test_split( \
X, y, test_size=0.2, random_state=1337)



#模型搭建
#此处需要神经网络基本知识，各个参数详细使用方法请参见https://keras.io/zh/
#神经元数量，根据需求设定
unit = 
#根据需求设定
batch_size=

#模型搭建，内含基本参数，根据需要设定，更多参见见上述网站
model = Sequential()
model.add(Dense(units=unit, input_dim= , activation= ,\
model.add(Dense(units=1, input_dim=unit, activation= ))

#模型编译，内含基本参数，根据需要设定，更多参见见上述网站
model.compile(loss= , optimizer= )

#训练
history = model.fit(X_train, y_train, epochs=200, \
batch_size=batch_size, validation_data=(X_test, y_test), \
verbose=2)

#导入模型
# del model
# model = load_model('bp_re_1_1.h5')

#存储模型
model.save('filename')

#模型训练loss对比图
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
#plt.savefig('12-1-dropout075-loss',dpi=600) #PNG文件
pyplot.show()
 
yhat11 = model.predict(X_test, batch_size=batch_size)
yhat = MaxMin_inverse(yhat11)
y_test = MaxMin_inverse(y_test)


#计算误差
ab = mean_absolute_error(y_test, yhat)
sq = sqrt(mean_squared_error(y_test, yhat))
print('Test RMSE ab: %.3f' % ab)
print('Test RMSE sq: %.3f' % sq)

pyplot.plot(yhat[0:100], label='predict')
pyplot.plot(y_test[0:100], label='real')
pyplot.legend()
#plt.savefig('12-6-stateful-consequence',dpi=600) #PNG文件
pyplot.show()

#计算相关系数
from math import sqrt

def multipl(a, b):
    sumofab = 0.0
    for i in range(len(a)):
        temp = a[i] * b[i]
        sumofab += temp
    return sumofab


def corrcoef(x, y):
    '''计算相关系数'''
    n = len(x)
    # 求和
    sum1 = sum(x)
    sum2 = sum(y)
    # 求乘积之和
    sumofxy = multipl(x, y)
    # 求平方和
    sumofx2 = sum([pow(i, 2) for i in x])
    sumofy2 = sum([pow(j, 2) for j in y])
    num = sumofxy - (float(sum1) * float(sum2) / n)
    # 计算皮尔逊相关系数
    den = sqrt((sumofx2 - float(sum1 ** 2) / n) * (sumofy2 - float(sum2 ** 2) / n))
    return num / den

print(corrcoef(yhat, y_test))




