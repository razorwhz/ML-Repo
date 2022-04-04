import matplotlib.pyplot as plt
import numpy as np
import random as rd
mat = "{:^6s}"  # print输出字符串格式设置
f = open(b'D:\ML\pokemon.csv', encoding="UTF-8")  # 读取csv文件
a = f.readline()
title = a.split(',')
title = title[0:31]
b = []
i = 0
sli = []
while a:
    a = f.readline()
    sli = a.split(',')
    b.append(dict(zip(title,sli[0:31])))
    i = i+1
at = []
total = []
defend = []
hp = []
ii = 0
while ii < len(b)-1:
    at.append(float(b[ii]['attack']))
    total.append(float(b[ii]['base_total']))
    defend.append(float(b[ii]['defense']))
    hp.append(float(b[ii]['hp']))

    ii = ii+1

pool = []
ii = 0
while ii<len(b)-1:
    cur = []
    cur.append(at[ii])
    cur.append(defend[ii])
    cur.append(hp[ii])
    cur.append(total[ii])
    pool.append(cur)
    ii = ii+1
print(pool[10:20])
def mini_batch(LIST,batch_size):
    pool_size = np.array(range(0,len(LIST)))
    np.random.shuffle(pool_size)
    batch_number = list(pool_size)
    batchN = []
    while len(batch_number)>batch_size:
        batchN.append(batch_number[0:batch_size])
        del batch_number[0:batch_size]
    batchN.append(batch_number)
    return np.array(batchN)
def initial_network(layer_dims):
    """
	:param layer_dims: list,每一层单元的个数（维度）
	:return:dictionary,存储参数w1,w2,...,wL,b1,...,bL
	"""
    depth = len(layer_dims)
    para = {}
    for i in range(1,depth):
        para['W'+str(i)] = np.random.randn(layer_dims[i],layer_dims[i-1])
        para['b'+str(i)] = np.random.randn(layer_dims[i])
    return para

def sigmoid(Z,state = False):
    if state == False:
        output = 1/(1+np.exp(-Z))
    else:
        output = Z*(1-Z)
    return output

def RMS(result,LIST,state = False):
    size = len(LIST)
    sum = 0
    if state == False:
        for i in range(0,size):
            sum = sum + 0.5*(np.array(LIST[i][len(LIST[0])-1])-result[0][i][len(result[0][i])-1])**2#len(result[0][i])-5
        return (sum/size)
    if state == True:
        for i in range(0,size):
            sum = sum + (np.array(LIST[i][len(LIST[0])-1])-result[0][i][len(result[0][i])-1])
        return (sum/size)
        



def forward_pro(LIST,size_dim,parameter):
    """
	:LIST: list,输入mini batch数据
    :size_dim: list,神经网络每一层的维度
	:return:array,每一层的输出值及最终结果
	"""
    size = len(size_dim)
    batch_size = len(LIST)
    result = []
    Zresult = []
    for i in range(0,batch_size):
        input = np.array(LIST[i][0:3])
        cur_result = [input]
        cur_Zresult = [0]
        for ii in range(1,size):
            cur_re = np.dot(parameter['W'+str(ii)],input)
            cur_Zresult.append(cur_re)
            if ii < size-1:
                cur_output = sigmoid(cur_re)
            else:
                cur_output = cur_re #要根据最后一层的激活函数进行调整
            cur_result.append(cur_output)
            input = cur_output
        result.append(cur_result)
        Zresult.append(cur_Zresult)
    return [result,Zresult] #将未经过激活函数的运算结果存入Zresult

def back_pro(result,parameter,LIST,layer_dims):
    size = len(layer_dims)
    batch_size = len(LIST)
    dev_W = list([0]*size)
    dev_b = list([0]*size)
    for ii in range(0,batch_size):
        partialW = list([0]*size)     #计算各层对Wi的偏微分
        partialb = list([0]*size)
        partialA = list([0]*size)
        partialZ = list([0]*size)
        for i in range(size-1,0,-1):
            if i == size-1:
                partialA[i] = RMS(result,LIST,True)
                partialZ[i] = partialA[i]
                partialW[i] = partialZ[i]*np.array(result[0][ii][i-1]) #ii是batch中的序号
                partialb[i] = partialZ[i]
                dev_W[i] = dev_W[i]+partialW[i]
                
                dev_b[i] = dev_b[i]+partialb[i]
                continue
            partialA[i] =  np.dot(parameter['W'+str(i+1)].T,partialZ[i+1])
            partialZ[i] =  partialA[i]*np.array(sigmoid(result[0][ii][i],True))
            nparray_result = np.array(result[0][ii][i-1]).reshape(-1,1)
            nparray_Z = partialZ[i].reshape(1,len(list(partialZ[i])))
            partialW[i] =  np.dot(nparray_Z.T,nparray_result.T)
            partialb[i] = partialZ[i]
            dev_W[i] = dev_W[i]+partialW[i]
            print(dev_W[i].shape) 
            dev_b[i] = dev_b[i]+partialb[i]
    return [np.array(dev_W)/batch_size,np.array(dev_b)/batch_size]
        

        
#每一隐藏层的w，a，b，z都要在反向传播分别计算，在进行梯度下降前求均值再乘learning rate



"""
testcase for forward_pro
"""
layer_dimention = [3,4,3,1]
parameter = initial_network(layer_dimention)
batchN = mini_batch(pool, 50)
batch_size = len(batchN)

for i in range(0, batch_size):
    cur_batch = []
    test_result = []
    test_Zresult = []
    for ii in batchN[i]:
       cur_batch.append(pool[ii])
    batch_result = forward_pro(cur_batch, layer_dimention, parameter)
    dev_result = back_pro(batch_result,parameter,cur_batch,layer_dimention)
    for i in range(1,len(layer_dimention)):
        parameter['W'+str(i)] = parameter['W'+str(i)]+0.001*dev_result[0][i]
        parameter['b'+str(i)] = parameter['b'+str(i)]+0.001*dev_result[1][i]
    







    

    




    
    




