# https://github.com/maniraman-periyasamy/QPattern/blob/206ee4c813c0de9f0d97e6f47b5eca044b567274/pattern.py
"""
This program implements a Quantum pattern search algorithm presented in the paper

Probabilistic Quantum Memories
by
C. A. Trugenberger

"""


from qiskit import *
import matplotlib.pyplot as plt
from qiskit import tools
from qiskit.tools.visualization import plot_histogram, plot_state_city
from qiskit.circuit.library import MCMT, MCXGate, Measure
from qiskit.extensions import unitary, UnitaryGate
from qiskit.quantum_info.operators import Operator
import numpy as np



class Q_Pattern:

    def __init__(self, pattern, n, m):

        self.pattern = pattern

        self.m = m
        self.n = n

        self.main_pR = QuantumRegister(self.n, "p")
        self.main_uR = QuantumRegister(2,"u")
        self.main_mR = QuantumRegister(self.n, "m")

        self.main_circuit = QuantumCircuit(self.main_pR, self.main_uR, self.main_mR)
        self.one_state = [0,1]
        self.zero_state = [1,0]

    def trainSuperPosition(self):

       
        for i in range(self.m):
            pR = QuantumRegister(self.n, "p")
            uR = QuantumRegister(2,"u")
            mR = QuantumRegister(self.n, "m")
            circuit = QuantumCircuit(pR,uR,mR, name="pattern"+str(i+1))

            for j in range(self.n):
                
                if self.pattern[i][j] == 0:
                    circuit.initialize(self.zero_state,pR[j])
                else:
                    circuit.initialize(self.one_state,pR[j])
                
                circuit.ccx(pR[j],uR[1],mR[j])

            for j in range(self.n):

                circuit.cx(pR[j],mR[j])
                circuit.x(mR[j])
                
            circuit.mcx(mR,uR[0])
                
            
            k = i+1
            data = np.array([[np.sqrt((k-1)/k),np.sqrt(1/k)],[-np.sqrt(1/k),np.sqrt((k-1)/k)]])
            gate = UnitaryGate(data=data)
            gate = gate.control(1,ctrl_state="1")
            circuit.append(gate,[uR[0],uR[1]],[])

            circuit.mcx(mR,uR[0])


            for j in range(self.n):

                circuit.x(mR[j])
                circuit.cx(pR[j],mR[j])

            for j in range(self.n):
                circuit.ccx(pR[j],uR[1],mR[j])
            
            """circuit.draw(output = "mpl")
            plt.tight_layout()
            plt.show()"""
            self.main_circuit.append(circuit.to_instruction(), self.main_pR[:self.n]+ self.main_uR[:2] + self.main_mR[:self.n])
        
        self.main_circuit.draw(output = "mpl")
        plt.tight_layout()
        plt.show()
        return self.main_circuit


    def retrive(self, x):
        
        """
        To be implemented
        """
        



pattern = [[1,0]]
x = [1,0]
Q_obj = Q_Pattern(pattern,2,1)
Q_obj.trainSuperPosition()


    