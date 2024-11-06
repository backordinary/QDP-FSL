# https://github.com/vardhannegi/Task1/blob/ea7e8628c0a7ba86ddc52290d3056ccc8a062c7e/inputDatagen.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute
from numpy import linalg as LA
import pandas as pd
import numpy as np
import pprint
import random
import time

ep = []
layer_count = 0
final_theta ={}
circuitlist = []
simulator = Aer.get_backend('statevector_simulator')
y_upper = int((8+2)/2)
theta_range = np.linspace(0, 2 * np.pi, 128)
random_state = np.array([1,0,1,1,1,0,0,0,0,0,1,1,0,1,0,0])

class circuitgen:
    def __init__(self,total_layer,total_instance):
        self.n = total_layer 
        self.m = total_instance   
            
        for x in range(1,self.m+1):
            
            theta_dict = {f'instance{x}': {}}
            for y in range(1,8*self.n+1):
              theta_dict[f'instance{x}']['{0}'.format(str(y).zfill(2))] = random.choice(theta_range)
              final_theta.update(theta_dict)
              
            cap = 0
            qc = f'qc{x}'
            qc = QuantumCircuit(4,4)
            
            for _ in range(1,self.n+1):
              for y in range(1,y_upper):
                zz = cap + y
                a = '{0}'.format(str(zz).zfill(2))
                # print(a)
                qc.rx(final_theta[f'instance{x}'][a],y-1)

              for num in range(4):
                if num<=2:
                  qc.cz(0,num+1)

              for num in range(4):
                if num<2:
                  qc.cz(1,(num+1)+1)
              qc.cz(2,3)
              for y in range(y_upper,8+1):
                if y <8:
                  zz = cap + y
                  a = '{0}'.format(str(zz).zfill(2))
                  # print(a)
                  qc.rx(final_theta[f'instance{x}'][a],y-5)
                else :
                  zz = cap + y
                  a = '{0}'.format(str(zz).zfill(2))
                  # print(a)
                  qc.rx(final_theta[f'instance{x}'][a],y-5)
                  cap = zz
              qc.barrier()
            
            circuitlist.append(qc)
##           return circuitlist
          
    def getInputData(self):
        for x in circuitlist:
              result = execute(x, simulator).result()
              statevector = result.get_statevector(x)
              epsilon = LA.norm(statevector - random_state)
              ep.append(epsilon)
        df = pd.DataFrame.from_dict(final_theta)
        kk = df.T
        kk['epsilon'] = ep 
        return kk
      

##kk.to_csv('2x10kmodel.csv')
