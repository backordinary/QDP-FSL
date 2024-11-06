# https://github.com/naomisolomons/qflow/blob/be3a1aa29d926d1a7e49aa546567b60df7dad286/Qflow_Level_Builder.py
import sys; sys.executable
from qiskit.quantum_info import Operator
from qiskit.quantum_info import Statevector
import numpy as np
#This Module takes in a Qiskit QuantumCircuit object and constuctes a stage 
#object. This Stage object contains all the levels of the stage and the 
#state of the system after each level. 

class stage:
    
    def __init__(self, Qiskit_circuit): #takes input of Qiskit Circuit
        self.circuit = Qiskit_circuit
        self.num_levels = (Qiskit_circuit.depth()) #define the number of levels as the depth of the circuit.
        self.array = np.zeros(Operator(self.circuit).data.shape[0]) #prepare an empty state vector with the dimension of the system
        self.array[0] = 1 
        self.initial = Statevector(self.array) # this sets the initial state of the system to be the all zeros string.
        
    def reduce(self,circ): #this method takes a circuit and removes gates until the depth decreases by one. 
        Initial_depth = circ.depth()
        depth = circ.depth()
        data = circ.data
        while depth == Initial_depth:
            data.pop()
            depth = circ.depth()
        return circ
        
    def subcirc(self,Level_Num): # Say we have a circuit of depth N and we want a circuit of depth D, then this method reduces the initial circuit until it has a depth = D
        reduction_val = self.num_levels - Level_Num
        subcirc = self.circuit.copy()
        for i in range(reduction_val):
            subcirc = self.reduce(subcirc)
        return subcirc    
        
    def level(self,Level_Num): # This meothd the constructs the nth level as the difference between circuits of depth d and d-1
        Subcirc_N = self.subcirc(Level_Num)
        Subcirc_N_0 = self.subcirc(Level_Num-1)
        data = Subcirc_N.data
        data_0 = Subcirc_N_0.data
        Subcirc_N.data = [item for item in data + data_0 if item not in data or item not in data_0]       
        return Subcirc_N
    
    def levelOperator(self,Level_Num):  #Gives the unitary representation of level =Level_Num
        QC = self.level(Level_Num)
        leveloperator = Operator(QC) 
        return leveloperator
    
    def levelState(self,Level_Num): #gives the state of the system at the start of level = Level_Num
        levelstate = self.initial
        for i in range(1,Level_Num+1):
            levelstate = levelstate.evolve(self.levelOperator(i))
        return levelstate

#This way of constructing levels takes a circuit of depth d and gives d levels. 
#This is the smallest partitioning without going to the single gate level.
#This choice is arbitary and could be changed if necessary to give levels of arbitary depth. 
#This can be simply done by changing the condition in the while loop used in the reduce method. 
#Although one must tkae care if (depth/Level Number) is not a whole number.         


if __name__== '__main__': #Testing code
    from qiskit import QuantumCircuit
    
    qs = QuantumCircuit(5)
    qs.h((0,1))
    qs.x((0,1))
    qs.cx(0,4)
    stage = stage(qs)
    print(stage.circuit)
    print(stage.level(1))
    print(stage.level(2))
    print(stage.level(3))
    
