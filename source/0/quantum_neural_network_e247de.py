# https://github.com/jamesjcai/quantum/blob/0f00afba4feea07b53b18d3057b1af2e1306443f/quantum_neural_network.py

from qiskit import QuantumCircuit, Aer, IBMQ, QuantumRegister, ClassicalRegister, execute
from sklearn import model_selection, datasets, svm
import numpy as np
import qiskit
import copy
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X=iris.data[0:100]
Y=iris.target[0:100]
X_train,X_test,Y_train,Y_test=model_selection.train_test_split(X,Y,test_size=0.33,random_state=42)


N=4
def feature_map(X):
    q=QuantumRegister(N)
    c=ClassicalRegister(1)
    qc=QuantumCircuit(q,c)
    for i, x in enumerate(X):
        qc.rx(x,i)
    return qc, c

def variational_circuit(qc, theta):
    for i in range(N-1):
        qc.cnot(i, i+1)
    qc.cnot(N-1,0)
    for i in range(N):
        qc.ry(theta[i],i)
    return qc



def quantum_nn(X, theta, simulator=True):
    qc,c=feature_map(X)

    qc=variational_circuit(qc,np.random.rand(N))
    qc.measure(0,c)
    shots=1E4
    backend = Aer.get_backend('qasm_simulator')

    job=qiskit.execute(qc, backend, shots=shots)
    result=job.result()
    counts=result.get_counts(qc)
    return counts['1']/shots

def loss(prediction,target):
    return (prediction-target)**2

def gradient(X,Y,theta):
    delta = 0.01
    grad =[]
    for i in range(len(theta)):
        dtheta=copy.copy(theta)
        dtheta[i] +=delta
        pred1=quantum_nn(X,dtheta)
        pred2=quantum_nn(X,theta)
        grad.append((loss(pred1,Y)-loss(pred2,Y))/delta)
    return np.array(grad)

def accuracy(X,Y,theta):
    counter=0
    for X_i,Y_i in zip(X,Y):
        prediction = quantum_nn(X_i,theta)
        if prediction<0.5 and Y_i==0:
            counter +=1
        elif prediction>=0.5 and Y_i ==1:
            counter +=1
    return counter/len(Y)

        

eta = 0.05
loss_list=[]
theta=np.ones(N)

print('Epoch \t Loss \t Training Accuracy')

for i in range(20):
    loss_tmp =[]
    for X_i, Y_i in zip(X_train,Y_train):
        prediction=quantum_nn(X_i, theta)
        loss_tmp.append(loss(prediction,Y_i))
        theta = theta -eta * gradient(X_i, Y_i, theta)

    loss_list.append(np.mean(loss_tmp))
    acc = accuracy(X_train,Y_train,theta)

    print(f'{i} \t {loss_list[-1]:.3f} \t {acc:.3f}')




quantum_nn(X_train[5], np.random.rand(N))

plt.plot(loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


accuracy(X_test,Y_test,theta)


clf = svm.SVC()
clf.fit(X_train,Y_train)
print(clf.prediction(X_test))
print(Y_test)





