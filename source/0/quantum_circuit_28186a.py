# https://github.com/ioankolot/max_cut-aqc_pqc/blob/f7efc6cf1e32365e3cef6c6d158b235e1d800055/quantum_circuit.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.visualization import *
import numpy as np


class QCir():
    def __init__(self, number_of_qubits, thetas, connectivity, ansatz_family):

        self.number_of_qubits = number_of_qubits
        self.thetas = thetas    

        self.qcir = QuantumCircuit(self.number_of_qubits)


        if ansatz_family == "rycxrz":
            for qubit in range(self.number_of_qubits):
                self.qcir.ry(self.thetas[qubit], qubit)

            self.qcir.barrier()

            #We allow three different cases for the connectivity. nearest-neighbors, all-to-all
            if connectivity == 'nearest-neighbors':
                for qubit1 in range(self.number_of_qubits-1):
                    self.qcir.cnot(qubit1, qubit1+1)
                    
                self.qcir.cnot(self.number_of_qubits-1, 0)
            

            elif connectivity == 'all-to-all':
                for qubit1 in range(self.number_of_qubits):
                    for qubit2 in range(self.number_of_qubits):
                        if qubit1<qubit2:
                            self.qcir.cnot(qubit1, qubit2)


            self.qcir.barrier()

            for qubit in range(self.number_of_qubits):
                self.qcir.rz(self.thetas[qubit + self.number_of_qubits], qubit)

        elif ansatz_family == "ryczry":
            for qubit in range(self.number_of_qubits):
                self.qcir.ry(self.thetas[qubit], qubit)

            self.qcir.barrier()

            if connectivity == 'nearest-neighbors':
                for qubit1 in range(self.number_of_qubits-1):
                    self.qcir.cz(qubit1, qubit1+1)
                    
                self.qcir.cz(self.number_of_qubits-1, 0)
            

            elif connectivity == 'all-to-all':
                for qubit1 in range(self.number_of_qubits):
                    for qubit2 in range(self.number_of_qubits):
                        if qubit1<qubit2:
                            self.qcir.cz(qubit1, qubit2)

            self.qcir.barrier()

            for qubit in range(self.number_of_qubits):
                self.qcir.ry(self.thetas[qubit + self.number_of_qubits], qubit)


        elif ansatz_family == 'rycxrzcxrz':
            for qubit in range(self.number_of_qubits):
                self.qcir.ry(self.thetas[qubit], qubit)

            self.qcir.barrier()


            for qubit1 in range(self.number_of_qubits-1):
                self.qcir.cx(qubit1, qubit1+1)
                        
            self.qcir.cx(self.number_of_qubits-1, 0)
                

            self.qcir.barrier()

            for qubit in range(self.number_of_qubits):
                self.qcir.rz(self.thetas[qubit + self.number_of_qubits], qubit)

            for qubit1 in range(self.number_of_qubits-1):
                self.qcir.cx(qubit1, qubit1+1)
                        
            self.qcir.cx(self.number_of_qubits-1, 0)

            for qubit in range(self.number_of_qubits):
                self.qcir.rz(self.thetas[qubit + 2*self.number_of_qubits], qubit)


        elif ansatz_family == 'rycxrzcxrzcxrz':
            for qubit in range(self.number_of_qubits):
                self.qcir.ry(self.thetas[qubit], qubit)

            self.qcir.barrier()


            for qubit1 in range(self.number_of_qubits-1):
                self.qcir.cx(qubit1, qubit1+1)
                        
            self.qcir.cx(self.number_of_qubits-1, 0)
                

            self.qcir.barrier()

            for qubit in range(self.number_of_qubits):
                self.qcir.rz(self.thetas[qubit + self.number_of_qubits], qubit)

            for qubit1 in range(self.number_of_qubits-1):
                self.qcir.cx(qubit1, qubit1+1)
                        
            self.qcir.cx(self.number_of_qubits-1, 0)

            for qubit in range(self.number_of_qubits):
                self.qcir.rz(self.thetas[qubit + 2*self.number_of_qubits], qubit)

            for qubit1 in range(self.number_of_qubits-1):
                self.qcir.cx(qubit1, qubit1+1)
                        
            self.qcir.cx(self.number_of_qubits-1, 0)

            for qubit in range(self.number_of_qubits):
                self.qcir.rz(self.thetas[qubit + 3*self.number_of_qubits], qubit)



        elif ansatz_family == 'ryczryczry':

            for qubit in range(self.number_of_qubits):
                self.qcir.ry(self.thetas[qubit], qubit)

            self.qcir.barrier()

            for qubit1 in range(self.number_of_qubits):
                for qubit2 in range(self.number_of_qubits):
                    if qubit1 < qubit2:
                        self.qcir.cz(qubit1, qubit2)
                        
                
            self.qcir.barrier()

            for qubit in range(self.number_of_qubits):
                self.qcir.ry(self.thetas[qubit + self.number_of_qubits], qubit)

            self.qcir.barrier()
            
            for qubit1 in range(self.number_of_qubits):
                for qubit2 in range(self.number_of_qubits):
                    if qubit1 < qubit2:
                        self.qcir.cz(qubit1, qubit2)


            self.qcir.barrier()

            for qubit in range(self.number_of_qubits):
                self.qcir.ry(self.thetas[qubit + 2*self.number_of_qubits], qubit)






