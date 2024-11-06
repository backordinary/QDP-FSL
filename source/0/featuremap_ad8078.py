# https://github.com/fatginger1024/PQC_Descriptor/blob/ebc57c542eb5448e8e4c0e1ff07637a4a9e4bc5f/pqc/featuremap.py

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


class FeatureMap:
    
    def __init__(self,qubits:int=4):
        self.qubits = qubits
        
        
    def circuit1(self):
        
        qc = QuantumCircuit(self.qubits)
        # create all parameters inside the qc 
        x = ParameterVector(r'$\theta$', length=8)
        # add parametrised gates
        [qc.rx(x[int(2*i)], i) for i in range(self.qubits)]
        # add parametrised gates
        [qc.rz(x[int(2*i+1)], i) for i in range(self.qubits)]
        
        
        return qc
    
    def circuit2(self):
        qc = QuantumCircuit(self.qubits)
        # create all parameters inside the qc 
        x = ParameterVector(r'$\theta$', length=8)
        # add parametrised gates
        [qc.rx(x[int(2*i)], i) for i in range(self.qubits)]
        # add parametrised gates
        [qc.rz(x[int(2*i+1)], i) for i in range(self.qubits)]
        qc.cx(range(1, self.qubits), range(self.qubits-1))
        
        return qc
    
    def circuit3(self):
        qc = QuantumCircuit(self.qubits)
        # create all parameters inside the qc 
        x = ParameterVector(r'$\theta$', length=11)
        # add parametrised gates
        [qc.rx(x[int(2*i)], i) for i in range(self.qubits)]
        # add parametrised gates
        [qc.rz(x[int(2*i+1)], i) for i in range(self.qubits)]
        [qc.crz(x[8+i],i+1,i)for i in range(self.qubits-1)]
        
        return qc
    
    def circuit4(self):
        
        qc = QuantumCircuit(self.qubits)
        # create all parameters inside the qc 
        x = ParameterVector(r'$\theta$', length=11)
        # add parametrised gates
        [qc.rx(x[int(2*i)], i) for i in range(self.qubits)]
        # add parametrised gates
        [qc.rz(x[int(2*i+1)], i) for i in range(self.qubits)]
        [qc.crx(x[8+i],i+1,i)for i in range(self.qubits-1)]
        
        
        return qc
    
    def circuit5(self):
        
        qc = QuantumCircuit(self.qubits)
        # create all parameters inside the qc 
        x = ParameterVector(r'$\theta$', length=11)
        # add parametrised gates
        [qc.rx(x[int(2*i)], i) for i in range(self.qubits)]
        # add parametrised gates
        [qc.rz(x[int(2*i+1)], i) for i in range(self.qubits)]
        [qc.cry(x[8+i],i+1,i)for i in range(self.qubits-1)]
        
        return qc
    
    
    def circuit6(self):
        
        qc = QuantumCircuit(self.qubits)
        # create all parameters inside the qc 
        x = ParameterVector(r'$\theta$', length=8)
        # add parametrised gates
        [qc.ry(x[int(2*i)], i) for i in range(self.qubits)]
        # add parametrised gates
        [qc.rz(x[int(2*i+1)], i) for i in range(self.qubits)]
        qc.cx(range(1, self.qubits), range(self.qubits-1))
        
        return qc
    
    
    def circuit7(self):
        
        qc = QuantumCircuit(self.qubits)
        # create all parameters inside the qc 
        x = ParameterVector(r'$\theta$', length=11)
        # add parametrised gates
        [qc.ry(x[int(2*i)], i) for i in range(self.qubits)]
        # add parametrised gates
        [qc.rz(x[int(2*i+1)], i) for i in range(self.qubits)]
        [qc.crx(x[8+i],0,i+1)for i in range(self.qubits-1)]
        
        return qc
        
        
    def circuit8(self):
        
        qc = QuantumCircuit(self.qubits)
        # create all parameters inside the qc 
        x = ParameterVector(r'$\theta$', length=11)
        # add parametrised gates
        [qc.ry(x[int(2*i)], i) for i in range(self.qubits)]
        # add parametrised gates
        [qc.rz(x[int(2*i+1)], i) for i in range(self.qubits)]
        [qc.crz(x[8+i],i+1,i)for i in range(self.qubits-1)]
        
        return qc
    
    
    def circuit9(self):
        
        qc = QuantumCircuit(self.qubits)
        # create all parameters inside the qc 
        x = ParameterVector(r'$\theta$', length=11)
        # add hadamard gate to each qubit
        [qc.h(i) for i in range(self.qubits)] 
        # add parametrised gates
        [qc.ry(x[int(2*i)], i) for i in range(self.qubits)]
        # add parametrised gates
        [qc.rz(x[int(2*i+1)], i) for i in range(self.qubits)]
        [qc.crz(x[8+i],i+1,i)for i in range(self.qubits-1)]
        
        return qc
    
    
    def circuit10(self):
        
        qc = QuantumCircuit(self.qubits)
        # create all parameters inside the qc 
        x = ParameterVector(r'$\theta$', length=11)
        # add parametrised gates
        [qc.ry(x[int(2*i)], i) for i in range(self.qubits)]
        # add parametrised gates
        [qc.rz(x[int(2*i+1)], i) for i in range(self.qubits)]
        
        [qc.crz(x[8+i],i+1,i)for i in range(self.qubits-1)]
        
        return qc
    
    def circuit11(self):
        
        qc = QuantumCircuit(self.qubits)
        # create all parameters inside the qc 
        x = ParameterVector(r'$\theta$', length=8)
        # add hadamard gate to each qubit
        [qc.h(i) for i in range(self.qubits)] 
        [qc.cz(i+1,i)for i in range(self.qubits-1)]
        # add parametrised gates
        [qc.rx(x[int(2*i)], i) for i in range(self.qubits)]
        # add parametrised gates
        [qc.rz(x[int(2*i+1)], i) for i in range(self.qubits)]
        
        return qc
        
        
        
        
        
        