# https://github.com/SiyuanWuSFU/CMPT409-quantum-computing/blob/e6fad41c8c9482bd7e061c6e44b1c878d1b1e96c/qmlcx.py
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import math
import sklearn.datasets as datasets
from scipy.optimize import minimize 
from functools import partial
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, BasicAer, IBMQ
from sklearn import preprocessing
import matplotlib as mpl

import matplotlib.pyplot as plt

def show_figure(fig):
    new_fig = plt.figure()
    new_mngr = new_fig.canvas.manager
    new_mngr.canvas.figure = fig
    fig.set_canvas(new_mngr.canvas)
    plt.show(fig)
def initial_state(qubit_size,data):
    qr = QuantumRegister(qubit_size)
    cr = ClassicalRegister(qubit_size)
    neural_circit = QuantumCircuit(qr,cr)
    for qubit in range(qubit_size):
       neural_circit.h(qubit)
    
    for i in range(len(data)):
        neural_circit.rz((-math.pi*data[i]),i)



      
    ##print(neural_circit)
    return neural_circit

def Uloc(neural_circit,theta_y,theta_z):#theta_z is a 1x4 matrix
    ##print(neural_circit.num_qubits)
    for i in range(neural_circit.num_qubits):
        neural_circit.ry(-theta_y[i],i)
        neural_circit.rz(-theta_z[i],i)
   ## print(neural_circit)
    return neural_circit

def Uent(neural_circit): 
    for i in range(neural_circit.num_qubits -1):
        neural_circit.cx(i,i+1)
         
    neural_circit.cx(0,neural_circit.num_qubits -1)
    neural_circit.barrier()

    ##print(neural_circit)
    return neural_circit

def measurement(probability,y):                                 ##return probability sum for each case 
    freq_y=0
    if y==0: 
        for i in probability:
            if i=='0001' or i=='0010' or i=='0100' or i=='1000':
                ##print(result[i])
                freq_y+=probability[i]
        ##print(freq_y)

    if y==1:
        for i in probability:
            if i=='1110' or i=='1101' or i=='1011' or i=='0111': 
                freq_y+=probability[i]

    if y==2: 
        for i in probability: 
            if i=='0011' or i=='0110' or i=='1100' or i=='1001': 
                freq_y+=probability[i]
    
    return freq_y

def loss(probability,y,label_size): 
    freq_total=0
    freq=[]
    for i in range(label_size): 
        freq.append(measurement(probability,i))
        freq_total+=measurement(probability,i)
    freq_without_y=list(freq)
    del freq_without_y[y]
    loss_value =1/(1+math.exp(-(math.sqrt(freq_total)*(max(freq_without_y)-freq[y])/math.sqrt((freq_total-freq[y])*freq[y]))))
    return loss_value


def cost(theta,x,y,qubit_size,depth,label_size):                ##quantum part starts here 
    cost_value=0
    theta_y=np.zeros((depth,qubit_size))                         ##initialize 2-d array with size (depth x [0,0,0,0])
	
    theta_z=np.zeros((depth,qubit_size))
    for i in range(depth):
        for j in range(qubit_size): 
            theta_y[i][j]=theta[2*i*qubit_size+2*j]              ## theta_y = even postion of minimized theta
	
            theta_z[i][j]=theta[2*i*qubit_size+2*j+1]            ## theta_x = odd postion of minimized theta
    
    backend = BasicAer.get_backend('qasm_simulator')

    for i in range(len(y)): 
        NNC = initial_state(qubit_size,x[i])
        Uloc(NNC,theta_y[0],theta_z[0])
        for j in range(1,depth):
            Uent(NNC)
            Uloc(NNC,theta_y[j],theta_z[j])
        NNC.measure(0,0)     
        NNC.measure(1,1)
        NNC.measure(2,2)
        NNC.measure(3,3)
        probability = execute(NNC, backend, shots=1000).result().get_counts()       
        
        ##diagram = NNC.draw(output='mpl')
        ##show_figure(diagram)
        
        cost_value+=loss(probability,y[i],label_size)
    cost_value = cost_value / len(y)
    print(cost_value)
    
    return cost_value

def test(theta,x,qubit_size,depth,label_size):
    y_predict=[] 
    theta_y=np.zeros((depth,qubit_size))
    theta_z=np.zeros((depth,qubit_size))
    for i in range(depth):
        for j in range(qubit_size):
            theta_y[i][j]=theta[2*i*qubit_size+2*j]
            theta_z[i][j]=theta[2*i*qubit_size+2*j+1]
    for i in range(len(x)):
        NNC = initial_state(qubit_size,x[i])
        Uloc(NNC,theta_y[0],theta_z[0])

        for j in range(1,depth):
            Uent(NNC)
            Uloc(NNC,theta_y[j],theta_z[j])

        backend = BasicAer.get_backend('qasm_simulator')
        NNC.measure(0,0)
        NNC.measure(1,1)
        NNC.measure(2,2)
        NNC.measure(3,3)
        probability = execute(NNC, backend, shots=1000).result().get_counts()
        freq=[]
        for j in range(label_size):
            freq.append(measurement(probability,j))
        for j in range(label_size):
            if freq[j]==max(freq):
                y_predict.append(j)
                break

    return y_predict
def function(x,y,qubit_size,depth,label_size):                            ##create function for minimize 
    return partial(cost,x=x,y=y,qubit_size=qubit_size,depth=depth,label_size=label_size) ## call cost function 









def randomize_data(data,label,percentage): 
    data_size = len(data)                                               ## n_data = total number of sample label 
    permutation = np.random.permutation(range(data_size))                ## create an array range from 0-149

    test_size = int(data_size*percentage)                                  ## using 150 samples *0.8 =120
    train_size = data_size-test_size                                           ## n_test = 30

    train_permutation = permutation[0:train_size]                        ## random permutation 0-149	
    test_permutation = permutation[train_size:]                          ##random chose 30 numbers 
    ##print(len(test_permutation))
    return data[train_permutation],label[train_permutation],data[test_permutation],label[test_permutation]

def success_rate(predict_Y,test_Y):
    success=0
    fail=0
    for i in range(len(predict_Y)):
        if(predict_Y[i]==test_Y[i]):
            success+=1
        else:
            fail+=1
    return success,fail         

qubit_size = 4                                                        ## there are 4 features
depth = 2                                                               
label_size = 3                                                       ## number of type 
data,label=datasets.load_iris(return_X_y=True) 

train_data,train_label,test_data,test_label=randomize_data(preprocessing.normalize(preprocessing.scale(data)),label,0.2) ## first two permuted data set, last two are set sample for testing
                                                                           ## third para is percentage of data used for testing 
theta=np.random.random_sample(2*qubit_size*depth)                   ## create an array of random float range from 0-1, array size = (2*qubit_size*depth) 

result=minimize(function(train_data,train_label,qubit_size,depth,label_size),theta,method='Powell') ## minimize function using powell with theta


    
y_predict=[]
theta=result.x 
y_predict=test(theta,test_data,qubit_size,depth,label_size)
success,fail=success_rate(y_predict,test_label)
print("this is testing set: ", test_label) 
print("this is the predicting result: ", np.array(y_predict))
print(success/(success+fail))



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




