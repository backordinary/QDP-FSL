# https://github.com/braqiiit/QRealBoost/blob/09b364f55c08e18214871fe928cbe0cbd9ae9179/QAdaBoost/Breast%20Cancer/m32/qadaboostbc32.py
#!/usr/bin/env python
# coding: utf-8

# # Q Real Boost implementation 
# 
# New boosting implementation without QRAM, and with estimation and all for conference

# In[1]:


from qiskit import *
import math

import numpy as np
import pandas as pd
# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qiskit import *
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.tools.jupyter import *
from qiskit.visualization import *
#from ibm_quantum_widgets import *
from qiskit import BasicAer
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit import IBMQ
from sklearn.cluster import KMeans

"""### Helper Functions"""

def flip_string(x):
    return x[::-1]

def new_qc( mc, data):
    '''
    Unitary which is able to produce a number encoded in the binary format for production of k and b
    '''
    qr_m = QuantumRegister(mc)    
    
    qc = QuantumCircuit(qr_m, name = 'init')
    
    ## each of data points should be smaller than 2**mc
    
    for i in range(0,len(data)):
        if data[i]>2**mc:
            print("Error!! The value of the data to be stored is bigger then the 2**mc")
            return
    
    bin_data = ["" for x in range(len(data))]
    ## the data needs to convert to binary from decimal
    for i in range(0,len(data)):
        bin_data[i] = decimalToBinary(data[i], mc)
        
    
    new_data = np.zeros([len(data), mc])
    
    # now we will be dividing all our divided
    for i in range(len(data)):
        for j in range(mc):
            new_data[i, j] = bin_data[i][j]
    
    ## fliping the matrix around so the ordering is proper according QISKIT
    flip_new_data = np.flip(new_data,1)
    ## this will be arranged in a row vector so that we can run a loop over it 
    new_data_row = np.reshape(flip_new_data,[1,mc*len(data)])
    
    for i in range(len(new_data_row[0])):
        if new_data_row[0,i] == 1:
            qc.x(qr_m[i])
            
    return qc

def binaryToDecimal(binary):
     
    binary1 = binary
    decimal, i, n = 0, 0, 0
    while(binary != 0):
        dec = binary % 10
        decimal = decimal + dec * pow(2, i)
        binary = binary//10
        i += 1
    return decimal 
    
def numConcat(num1, num2): # this should actually do all the additions in the form of strings and then when you finally
                           # take out whatever is stored in the matrix then you should actually convert that to int
  
     # find number of digits in num2
    digits = len(str(num2))
    num2 = str(num2)
    num1 = str(num1)
  
     # add zeroes to the end of num1
#     num1 = num1 * (10**digits)
  
     # add num2 to num1
    num1 += num2
  
    return num1

## for convertign from decimal to binary 
def decimalToBinary(n,no_of_places):
    num = no_of_places ## this will be equal to mc
    binary = bin(n).replace("0b", "")
    if (len(binary) != num):
        i = num - len(binary)
        for j in range(0,i):
            binary = numConcat(0,binary)

    return binary

def dec_to_bin(n, size):
    bin_num = '0'*size
    b = bin(int(n)).replace("0b", "" )
    l = len(b)
    bin_num = bin_num[:size-l] + b
    return bin_num



def f2bin(number,places): 
    '''
    
    '''
    import numpy
    
    if number<0.0001:
        s = '.0000'
    else:

        if(type(number)==int or type(number)==numpy.int32 or type(number)==numpy.int64):

            return bin(number).lstrip("0b") + "."+"0"*places

        else:

            whole, dec = str(number).split(".") 
            whole = int(whole)
            dec = "0."+dec
            stri = ''
            res = bin(whole).lstrip("0b") + "."
            dec= float(dec)
            dec_val2 = dec
            num = dec
            countlen= 0

            while(dec_val2 != 0 and countlen <= places):

                num = float(num)*2
                arr = str(num).split(".")

                if (len(arr)==2):
                    whole1 = arr[0]
                    dec_val = arr[1]
                else:
                    whole1 = arr[0]
                    dec_val = '0'

                if whole1 == '0':
                    stri = stri + '0'
                else:
                    stri = stri+ '1'


                dec_val2 = float(dec_val)
                num = '0.'+dec_val
                countlen = len(stri)

            if (len(stri)<= places):
                stri = stri + '0'*(places - len(stri))
            elif(len(stri)>= places):
                stri = stri[:places]
            else:
                stri = stri

            s = bin(whole).lstrip("0b")+'.'+stri

    return s

def rot_circuit():
    '''
    used for doing conditional rotations on the values of Dti and Dbkti's 
    '''
    # here only 5 qubits are required as we are storing the values of Dti in these 4 qubits only. 
    theta = 2#np.pi
    num_qubits = 5
    qc = QuantumCircuit(num_qubits, name = 'rot_circuit')
    qc.cry(theta/2,0,4)
    qc.cry(theta/4,1,4)
    qc.cry(theta/8,2,4)
    qc.cry(theta/16,3,4)


    return qc


"""## Dataset - Trying out with Cancer prediction"""

import sklearn
from sklearn import svm

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier



## from now on we will 
X = np.array([[-0.52961818],
       [-0.73694129],
       [-0.98076452],
       [ 0.30469623],
       [-0.78349866],
       [-0.48788974],
       [-0.79870995],
       [ 0.25658932],
       [-0.70611322],
       [-0.74200188],
       [-0.7363877 ],
       [ 0.31199637],
       [-0.97364873],
       [ 0.71250752],
       [-0.55861402],
       [-0.63612516],
       [-0.90847131],
       [-0.54241534],
       [-0.21227206],
       [-0.49821623],
       [-0.7787262 ],
       [-0.69581626],
       [-0.77763277],
       [ 0.38337651],
       [-0.02174895],
       [-1.        ],
       [-0.63179088],
       [-0.9240283 ],
       [-0.83105823],
       [-0.61640269],
       [-0.95954346],
       [-0.81083621]])
y = np.array([1,
 1,
 1,
 0,
 1,
 1,
 0,
 0,
 1,
 1,
 1,
 0,
 1,
 0,
 0,
 1,
 1,
 1,
 0,
 0,
 1,
 1,
 1,
 0,
 0,
 1,
 1,
 1,
 1,
 1,
 1,
 1])


X_test = np.array([[-0.40839483],
       [-0.3025159 ],
       [-0.56281703],
       [ 1.        ],
       [-0.51608625],
       [-0.30182866],
       [-0.89821949],
       [-0.3258409 ],
       [ 0.24403701],
       [-0.08547389],
       [-0.81397378],
       [-0.52275241],
       [-0.84506346],
       [ 0.06944435],
       [ 0.12100471],
       [-0.82295834],
       [-0.56187142],
       [-0.47338434],
       [ 0.27271194],
       [-0.40050862],
       [-0.44610167],
       [-0.69609962],
       [-0.82142116],
       [-0.79394096],
       [-0.73949103],
       [-0.27940424],
       [ 0.80421303],
       [-0.32988797],
       [-0.78661245],
       [-0.75657348],
       [-0.77425805],
       [-0.89459281]])

y_test = np.array([1,
 0,
 0,
 0,
 0,
 0,
 1,
 0,
 0,
 0,
 1,
 0,
 0,
 0,
 0,
 1,
 0,
 1,
 0,
 0,
 1,
 1,
 1,
 1,
 1,
 1,
 0,
 0,
 1,
 1,
 1,
 1])




# here we will also define the vallue of Q
no_of_Q = 4

def get_ht_new(X, y,Dti, no_of_Q):
    '''
    This function is used for returning the partitioning of the X's
    no_of_Q : represents the number of Q samples that must be choosen from all the M samples
    '''
    # we will extract the Q values with top Dti's
    Dti = np.array(Dti)
    ind_max = np.argpartition(Dti, -no_of_Q)[-no_of_Q:]

#     no of paritions = 2
    no_of_paritons = 2
    km = KMeans(
        n_clusters=no_of_paritons, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0)

    # now we will pass the corresponding X and parts with the Q samples to train the model 
    fitted_km = km.fit(X[ind_max])
    # prediction will be obtained for all the samples
    prediction = fitted_km.predict(X)
    d = prediction
    
    print('###### the old predictions', d)
    

    from scipy.stats import mode 
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, d)
    cm_argmax = cm.argmax(axis=0)
    if (cm_argmax[0] == cm_argmax[1]):
        correct_d = d
    else:
        correct_d = np.array([cm_argmax[i] for i in d])
    
    print('###### the new predictions', correct_d)

    return  correct_d, fitted_km


def original_acc(X,y, no_of_Q):
    '''
    Function is created for checkign the accuracy of the classifier in the first iteration
    '''
    Dti = np.full(len(X),1/len(X))
    dti = Dti
    
    preds, cls = get_ht_new(X, X, Dti, no_of_Q)
    
    dti0_0 = []
    dti0_1 = []
    dti1_0 = []
    dti1_1= []
    dti2_0 = []
    dti2_1 = []


    for i in range(len(preds)):
        if preds[i] == 0 and y[i] == 0:
            dti0_0.append(dti[i])
        if preds[i] == 0 and y[i] == 1:
            dti0_1.append(dti[i])
        if preds[i] == 1 and y[i] == 0:
            dti1_0.append(dti[i])
        if preds[i] == 1 and y[i] == 1:
            dti1_1.append(dti[i])
        if preds[i] == 2 and y[i] == 0:
            dti2_0.append(dti[i])
        if preds[i] == 2 and y[i] == 1:
            dti2_1.append(dti[i])
            
            
    if sum(dti0_0) >= sum(dti0_1):
        y0 = 0
    else:
        y0 = 1
    if sum(dti1_0) >= sum(dti1_1):
        y1 = 0
    else:
        y1 = 1
    if sum(dti2_0) >= sum(dti2_1):
        y2 = 0
    else:
        y2 = 1
        
        
    # what is the final y's
    final_y = []
    
    for i in range(len(preds)):
        if preds[i] == 0:
            final_y.append(y0)
        if preds[i] == 1:
            final_y.append(y1)
        if preds[i] == 2:
            final_y.append(y2)
            
    acc = metrics.accuracy_score(final_y, y)

    return acc



# the new function 

def Oh_Dbk_custom_new(qc, qr1, qr_list, data_dict):
    '''
    This oracle is used for entangling a particular value |Ki> with |xi>, by performing a transformation given by 
    
    |xi>|0> --> |xi>|Ki>
    
    this is used for encoding the |Dti>,|yi> and |jti>
    
    '''
    
    reg1_len = qr1.size
    reg2_len = len(qr_list)
    data_size = len(data_dict)

    # for application of mct we need an array which takes in all the qubits from qr1... [qr1[0],qr1[1],qr1[2]...]
    qr1_arr = []
    
    for i in range(reg1_len):
        qr1_arr.append(qr1[i])
    
    # application of the main gates that there are 
    
    for i in range(data_size):
        string1 = flip_string(list(data_dict.keys())[i])
        string2 = flip_string(list(data_dict.values())[i])

        # the main idea is that the oracle looks at the all the values of xi and the corresponding jti that it wanna 
        # attach, then it makes the state 11111..(of the qubit register storing xi) and applies a mct to make this control and 
        # target is applied on jti finally we apply X to make the states back to the original xi's
        
        for j in range(len(string1)):
            if string1[j] == '0':
                qc.x(qr1[j])
        
        for j in range(len(string2)):
            if string2[j] == '1':
                qc.mct(qr1_arr, qr_list[j])
        
        for j in range(len(string1)):
            if string1[j] == '0':
                qc.x(qr1[j])

def Oh_Dbk_custom_new_inv(qc, qr1, qr_list, data_dict):
    '''
    inverse of the function given above    
    '''
    
    reg1_len = qr1.size
    reg2_len = len(qr_list)
    data_size = len(data_dict)
    # for application of mct we need an array which takes in all the qubits from qr1... [qr1[0],qr1[1],qr1[2]...]
    qr1_arr = []
    
    for i in range(reg1_len):
        qr1_arr.append(qr1[i])
    
    # application of the main gates that there are 
    
    for i in range(data_size):
        string1 = flip_string(list(data_dict.keys())[len(X)-i-1])
        string2 = flip_string(list(data_dict.values())[len(X)-i-1])

        
        # the main idea is that the oracle looks at the all the values of xi and the corresponding jti that it wanna 
        # attach, then it makes the state 11111.. and applies a mct to make this control and target is applied on jti
        # finally we apply X to make the states back to the original xi's
        
        for j in range(len(string1)):
            if string1[j] == '0':
                qc.x(qr1[j])
        
        for j in range(len(string2)):
            if string2[j] == '1':
                qc.mct(qr1_arr, qr_list[j])
        
        for j in range(len(string1)):
            if string1[j] == '0':
                qc.x(qr1[j])

"""## Creating oracle(ht) for th iteration"""

from qiskit.algorithms import AmplificationProblem
from qiskit.algorithms import Grover
from qiskit.providers.aer import AerSimulator
backend = AerSimulator()

def A_qaa( Dti_asrt):
    '''
    This executes the A for the amplification circuit
    '''
    
    qr_xi = QuantumRegister(5, 'xi')
    qr_Dti = QuantumRegister(4, 'dti')
    qr_final_rot = QuantumRegister(1,  'final_rot')
    
    qc = QuantumCircuit(qr_xi, qr_Dti, qr_final_rot)#cr)
    
#     here we will just apply hadamards instead of QRAM 
    qc.h(qr_xi)
    
    ## now we are adding up all the Oh Dbk Customs in one - Dti 
    
    # list of all the qubits in order 
    listofqubits = []

    #dti
    for i in range(qr_Dti.size):
        listofqubits.append(qr_Dti[i])


    ## making a list in this order 
    listofydj={}
    # print('Dti_asrt')
    # print(Dti_asrt)
    # print(type(Dti_asrt[0]))
    # print('----------------')
    # print('this one', f2bin(Dti_asrt[0], qr_Dti.size))
    # this list will be passed to the Oh_Dbk_custom function for the encoding of the |xi>|Dti> function 
    for i in range(len(X)):
        listofydj[dec_to_bin(i, qr_xi.size)] = str(flip_string(f2bin(Dti_asrt[i], qr_Dti.size)[1:5]))
    print(listofydj)
    
    Oh_Dbk_custom_new(qc,qr_xi,listofqubits,listofydj)
    
    
    qc = qc.compose(rot_circuit(),[qr_Dti[0],qr_Dti[1],qr_Dti[2],qr_Dti[3],qr_final_rot[0]])
    
    qcinv=QuantumCircuit(qr_xi, qr_Dti, qr_final_rot)
    Oh_Dbk_custom_new_inv(qc,qr_xi,listofqubits,listofydj)
    
    
    return qc,qcinv

def amplification( Dti, reps):
    '''
    this will be able to actually be able to create the amplification circuit, run it and obtain the values of Dti after
    the amplification has been done!!
    '''

    Ainit,qcinv = A_qaa(Dti)
#     Ainit.measure_all()


    widthh = Ainit.width()
    # oracle
    qo = QuantumCircuit(widthh)
    qo.z(widthh - 1)
    oracle =  qo

    problem = AmplificationProblem(oracle, state_preparation=Ainit)
    
    n = Ainit.width()
    qc = QuantumCircuit(n)
    qc  = qc.compose(Ainit)

    G = problem.grover_operator

    
    for rep in range(reps):
        qc = qc.compose(G)
        
    
    cr = ClassicalRegister(qc.width() - 5) # here 4 is for the number of Dti's used +1 is for rot qubit
    qr = QuantumRegister(qc.width())
    qc_qaa = QuantumCircuit(qr, cr)

    # appendiing the amplification
    qc_qaa.append(qc, qr)

    # the qubits you wanna measure
    meas_qr = []
    meas_cr = []
    for i in range(qc.width() - 5):
        meas_qr.append(qr[i])
        meas_cr.append(cr[i]) 


    qc_qaa.measure(meas_qr,meas_cr)
    backend = AerSimulator()
    shots = 1500
    result = execute(qc_qaa, backend, shots = shots).result().get_counts()

    listofnew_dti = {}
    for i in range(len(result.keys())):

        listofnew_dti[(list(result.keys())[i])] = (list(result.values())[i]/shots)

    sorted_dict = listofnew_dti.keys()
    sorted_dict = sorted(sorted_dict) ## ascending order sorting of keys

    ## now this dictonary will contain the values of Dti in sorted ordering 
    dti_sort = []
    for i in range(len(listofnew_dti.keys())):
        dti_sort.append((listofnew_dti[sorted_dict[i]]))

        
    return dti_sort


def A(y, preds, Dti):
    '''
    This circuit will estitmate the partition label weigths(the 's). 
    
    '''
    w=Dti

    qr_xi = QuantumRegister(5, 'xi')
    qr_yi = QuantumRegister(1, 'yi')
    qr_Dti = QuantumRegister(4, 'dti')
    qr_ht = QuantumRegister(1, 'ht')
    
    qr_Dbk = QuantumRegister(4, 'dbk') # this is required as we're doing Dti amplification in seperate circuit

    qr_final_rot = QuantumRegister(1,  'final_rot')
    cr = ClassicalRegister(1)
    
    qc = QuantumCircuit(qr_xi,qr_yi, qr_Dti, qr_ht,qr_Dbk, qr_final_rot)
    
#     here we will just apply hadamards instead of QRAM 
    qc.h(qr_xi)
    
    ## now we are adding up all the Oh Dbk Customs in one - yi - Dti - ht
    
    # list of all the qubits in order 
    listofqubits = []
    
    #yi
    listofqubits.append(qr_yi[0])
    
    #dti
    for i in range(qr_Dti.size):
        listofqubits.append(qr_Dti[i])
    
    #ht
    listofqubits.append(qr_ht[0])
    

    ## making a list in this order look at ordering doucmentation for explanantion
    listofydj={}
    print(str(flip_string(f2bin(Dti[i] ,qr_Dti.size)))) 
    for i in range(len(X)):
        
        listofydj[dec_to_bin(i, qr_xi.size)] = str(dec_to_bin(preds[i],qr_ht.size)) + str(flip_string(f2bin(Dti[i] ,qr_Dti.size)[1:5])) +str(dec_to_bin(y[i],qr_yi.size))
    print(listofydj)
        

    
    Oh_Dbk_custom_new(qc,qr_xi,listofqubits,listofydj)
    
    # as we want to measure ht != yi
    qc.x(qr_ht)
    qc.mct([qr_Dti[0],qr_yi[0],qr_ht[0]],qr_Dbk[0])
    qc.mct([qr_Dti[1],qr_yi[0],qr_ht[0]],qr_Dbk[1])
    qc.mct([qr_Dti[2],qr_yi[0],qr_ht[0]],qr_Dbk[2])
    qc.mct([qr_Dti[3],qr_yi[0],qr_ht[0]],qr_Dbk[3])
    
    ## this will also have this part for making sure that the '00' state is also cared for 
    qc.x(qr_ht)
    qc.x(qr_yi)
    qc.mct([qr_Dti[0],qr_yi[0],qr_ht[0]],qr_Dbk[0])
    qc.mct([qr_Dti[1],qr_yi[0],qr_ht[0]],qr_Dbk[1])
    qc.mct([qr_Dti[2],qr_yi[0],qr_ht[0]],qr_Dbk[2])
    qc.mct([qr_Dti[3],qr_yi[0],qr_ht[0]],qr_Dbk[3])
    qc.x(qr_yi)
    
    ## end of copying - note all the htis and yis should be what they have been originally encoded to be 
    
    
    qc = qc.compose(rot_circuit(),[qr_Dbk[0],qr_Dbk[1],qr_Dbk[2],qr_Dbk[3],qr_final_rot[0]])
    

    return qc


"""For updation in QAdaBoost, we need $\alpha = ln(\sqrt(1-\epsilon_t)/\sqrt(\epsilon_t))$

From this estimation, I haven't thought of any option other than numerically getting 1- epsilon_t, as the estimation gives only the value of epsilon_t
"""

from qiskit.algorithms import EstimationProblem
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import AerSimulator
from qiskit.algorithms import IterativeAmplitudeEstimation
backend = AerSimulator()
quantum_instance_qae = QuantumInstance(backend,shots = 10)

def iteration_iqae( y, preds,Dti):
    '''
    This function executes the cirucit given above and uses IQAE on it to get the values of W+ and W- once these are obtained
    it returns the value of beta's 
    '''
    
    w = Dti
    
    wkb = []

    alpha_t = 0
    
    Zt = 0
        
    ## the iterative amplitude estimation
    iae = IterativeAmplitudeEstimation(
        epsilon_target=0.02,  # target accuracy
        alpha=0.05,  # width of the confidence interval
        quantum_instance=quantum_instance_qae)
    
        
    ## doing for label 0 
    qq = A(y, preds,Dti)
        
    problem0 = EstimationProblem( 
    state_preparation=qq,  # A operator
    objective_qubits=[qq.width()-1],  # we want to see if the last qubit is in the state |1> or not!!
    grover_operator=None)

    est_amp0 = iae.estimate(problem0).estimation

    '''
    Just doing Laplace correction here like this, lets see if it works
    Maybe the Yes/No subroutine can be circumvented with this smoothing 
    
    '''

    if(est_amp0!=0):
        wkb.append(est_amp0)
        den = est_amp0*8

    else:
        den = 1e-8 #smoothing
        # wn.append(den)
    
    # just in case epsilon t is going above 1 due to approximate values
    if den > 0.99:
        den = 0.99

#     alpha_t = np.log(np.sqrt(1-den)/np.sqrt(den))
    print("--------------------------------------------------------------------")
    print("epsilon_t = ", den)
    print("--------------------------------------------------------------------")

#     Zt = 2*np.sqrt((1-den)*den)
    return den


import time

def update_dti(X, Dti, y):
    '''
    Used for updation of Dti values using the updation rule 
    by using alpha_t values
    '''
    # print("----------------------------------------------------------")

    ## first we will  normalize the Dti
    
    #     Dti_norm = Dti/sum(Dti)
    # print("-----------------------")
    print("-> Dti_updated:")
    print(Dti)
    # print("-----------------------")
    Dti_norm = Dti

    
    Dti_arc_sin_sqrt = np.arcsin(np.sqrt(Dti_norm))


    # print("-----------------------")
    print("->  arcsin(sqrt(Dti)) :")
    print(Dti_arc_sin_sqrt)
    # print("-----------------------")
    
    ##################### amplification - and obtiaing the Ht ##########################
    T=25
    reps = 3*round(math.log10(T)*np.sqrt(len(X)))
    
    # we pass unormalized Dti to this as normalization will take place automatically inside it!
    time_start_amp = time.time()
    Dti_amp = amplification(Dti_arc_sin_sqrt, reps)
    time_end_amp = time.time()

    # print("-----------------------")
    print("->  Dti values obatined after amplification :")
    print(Dti_amp)
    # print("-----------------------")

    # running the classical classifier
    no_of_Q = 4
    # this gives us the values of partitions obtained after training weights with w_i
    preds,classifier = get_ht_new(X,y, Dti_amp, no_of_Q)
    
    print(preds)
    
#     print(original_distribution(preds,y,Dti_arc_sin_sqrt)) #testing 2
    
    ################## estimation ##################
    
    time_start_iqae = time.time()
    eps_t = iteration_iqae(y,  preds, Dti_arc_sin_sqrt)
    time_end_iqae = time.time()

    dti_up = []
    
    dtiii = Dti
    
    delta = 1/(10*(no_of_Q * T*T))
    
    #################### updation of Dti #####################
    Q = 4
    ## here we will be taking the 'yes'/'no' condition into account
    if eps_t >= (1-delta)/(64*Q*T*T):
        print('yes')
        alpha_t = np.log(np.sqrt(1-eps_t)/np.sqrt(eps_t))
        Zt = 2*np.sqrt((1-eps_t)*eps_t)
            
        for i in range(len(Dti)):
            if(y[i]==preds[i]):
                dti_up.append(dtiii[i]*np.exp(-alpha_t)/((1+2*delta)*Zt))

            else:
                dti_up.append(dtiii[i]*np.exp(alpha_t)/((1+2*delta)*Zt))

    else:
        print('no')
        alpha_t = np.log(np.sqrt((Q*T*T) - 1))
        Zt = 2*(np.sqrt((Q*T*T) - 1)/(Q*T*T))
            
        for i in range(len(Dti)):
            if(y[i]==preds[i]):
                dti_up.append(dtiii[i]*np.exp(-alpha_t)*(2 - 1/(Q*T*T))/((1+2/(Q*T*T))*Zt))

            else:
                dti_up.append(dtiii[i]*np.exp(alpha_t)*(1/(Q*T*T))/((1+2/(Q*T*T))*Zt))
    
    # print("--------------------------------------------------------------------")
    print("-> Zt", Zt)
    print("-> alpha_t", alpha_t)
    # print("--------------------------------------------------------------------")

    # there normalisation is not working 
    dti_00 = dti_up/sum(dti_up)

    print("->  Time for amplification", time_end_amp - time_start_amp )
    print("->  Time for Estimation", time_end_iqae - time_start_iqae )
    
    return dti_00, alpha_t, preds, classifier



def final_bin_predictions(preds_mat, alpha,X,y_mod,predt):
    
    '''
    used for calculating the value of H(x) from the beta values
    '''
    
    final_bin = []

    for i in range(len(X)):
        Hx = 0
        for t in range(len(alpha)):
            Hx = Hx + alpha[t]*preds_mat[t][i]
            
        if(Hx>0):
            final_bin.append(1)
        else:
            final_bin.append(-1)
            
    #### extra shit for checking 
    
#     predt = preds_mat[i]        

    inst_acc = metrics.accuracy_score(y_mod,predt)
    
    # print("Instantenous accuracy",inst_acc )
    # ######################
    
    acc = metrics.accuracy_score(final_bin, y_mod)
    
    
    return final_bin, acc


def complete_imp(num_iterations,X,y):
    
    '''
    The function which puts all the above functions together in order to obtain the accuracy and classifiers for the testing 
    
    '''  

    print("Input X", X)
    print("Input y", y)

    
    # all the arrays
    Dti = np.full(len(X),1/len(X))
    
    #change labels from 0,1 to -1,1
    y_mod = []
    for i in range(len(y)):
        if y[i]==0:
            y_mod.append(-1)
        else:
            y_mod.append(1)

    dti = []
    alpha = []
    accuracy_final = []
    classifiers_all = []
    preds_mat = []
    
    for itr in range(num_iterations):

        print('-------------------------------------------------------------------------------------------------')
        print('ITERATION - ', itr, '\n')

        time_prtitr_start = time.time()
        dtii0,alpha0,preds0, classifier0 = update_dti(X, Dti, y)
        
        print(dtii0)
        dti.append(dtii0)
        alpha.append(alpha0)
        
        ## taking the preds(ht) from 0,1 to -1,1 in order to fix the problem with all [1 1 1 1...1 ] here
        preds_1 = []
        for i in range(len(y)):
            if preds0[i]==0:
                preds_1.append(-1)
            else:
                preds_1.append(1)
        
        preds_mat.append(preds_1)
        ########
        classifiers_all.append(classifier0)
        print(alpha)
        final_bin, acc = final_bin_predictions(preds_mat, alpha,X, y_mod,preds_1)
        print("New Binary labels : ", final_bin)
        print("New Accuracy : ", acc)
        accuracy_final.append(acc)
        
        Dti = dtii0

        time_prtitr_end = time.time()
        
        print("->  Total time per iteration", time_prtitr_end - time_prtitr_start)

    
    import matplotlib.pyplot as plt
    from matplotlib import style
    
    plt.style.use('seaborn')
    plt.plot(list(range(num_iterations)), accuracy_final)
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.show()
    
    return dti, accuracy_final, alpha, classifiers_all



"""#**14/05/22**"""

no_of_itr = 25
dti, accuracy_final, alpha, classifiers_all= complete_imp(no_of_itr,X,y)

print("-------------------")
print("Training accuracy :" , accuracy_final)
print("-------------------")
print("alpha :" , alpha)
print("-------------------")
print("Final dti :" , dti)
print("-------------------")


"""# Testing """


print('Testing Data : ')
print(X_test)
print(y_test)

def Testing(classifiers, alpha_t,X,y, No_of_itr):
    
    ## all we need to do is to do prediction with the earlier trained classifiers
    ht = []
    accuracy_final_test = []
    pred=[]
    
    #change labels from 0,1 to -1,1
    h_mod = []
    
    y_mod = []
    for i in range(len(y)):
        if y[i]==0:
            y_mod.append(-1)
        else:
            y_mod.append(1)
            
            
    for t in range(No_of_itr):

        d = classifiers[t].predict(X)
        # ht.append(d)
        from scipy.stats import mode 
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y, d)
        cm_argmax = cm.argmax(axis=0)
        if (cm_argmax[0] == cm_argmax[1]):
            correct_d = d
        else:
            correct_d = np.array([cm_argmax[i] for i in d])
        
        print('-> Corrected predictions', correct_d)
        ht.append(correct_d)
        
        for i in range(len(ht[0])):
            if ht[t][i]==0:
                ht[t][i] = -1
            else:
                ht[t][i] = 1
    
    # now ht has all the particular values

    for t in range(No_of_itr):
        pred.append(alpha[t]*ht[t])
    print(pred)
    
    
    for t in range(No_of_itr):  
        final_pred=[]
        # adding up all the alpha*ht
        predsum = np.sum(pred[0:t+1], axis = 0)
        print(predsum)
        
        for i in range(len(X)):
            if predsum[i]>0:
                final_pred.append(1)
            else:
                final_pred.append(-1)

        print('-> Final Predictions: ',final_pred)
        accuracy_final_test.append(metrics.accuracy_score(y_mod,final_pred))

    import matplotlib.pyplot as plt
    from matplotlib import style
    
    plt.style.use('seaborn')
    plt.plot(list(range(No_of_itr)), accuracy_final_test)
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Breast Cancer Testing Accuracy M=8")
    plt.show()
        
    print('Testing Accuracy : ', accuracy_final_test)    
    return final_pred

final_pred = Testing(classifiers_all, alpha,X_test,y_test,25)


print("-------------------")
print("Final Testing accuracy", final_pred)
print("-------------------")

