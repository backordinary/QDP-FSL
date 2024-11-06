# https://github.com/justids/Predicting-ground-state-with-novel-quantum-descriptor-of-molecules-using-QML-/blob/a9e1263b08df878bd637c35070a5cfd1b144bf17/2022Qiskit_QNNP.py
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import transpile, assemble
from qiskit.visualization import *
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile, assemble
from qiskit import Aer
from qiskit.tools.visualization import circuit_drawer
from qiskit.quantum_info import state_fidelity
from qiskit import BasicAer
from qiskit.circuit.library import TwoLocal
from AtomLoader import AtomLoader 



class QuantumCircuit:
    """ 
    This class provides a simple interface for interaction 
    with the quantum circuit 
    """
    
    def __init__(self, n_qubits, depth, backend):

        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.n_qubits=n_qubits
     
        self.ksi = qiskit.circuit.Parameter('ksi')
        twolocal=TwoLocal(num_qubits=n_qubits, reps=depth, rotation_blocks=['ry','rz'], 
                   entanglement_blocks='cx', entanglement='circular', parameter_prefix='ξ', insert_barriers=True)
        twolocal=self.twolocal.bind_parameters(self.ksi)
        self._circuit.barrier()
        self._circuit+=twolocal
        self._circuit.barrier()
        self._circuit.z(0)
        self._circuit.barrier()
        self._circuit+=twolocal.inverse()
        self._circuit.barrier()

        self.backend = backend
    
    def run(self, parameters, idx):
        list_eta = parameters[0:self.n_qubits]
        list_ksi = parameters[self.n_qubits:]
        atom_data = AtomLoader(list_eta, idx)
        ground_energy_label = atom_data[idx]['ground_energy'][0]/-20000
        descriptors = atom_data[idx]['descriptor']
        n_atoms = len(atom_data[idx]['descriptor'])
        
        energy=0
        for i, descriptor in enumerate(descriptors):
            qc_descriptor = qiskit.QuantumCircuit(self.n_qubits)
            for j, description in enumerate(descriptor):
                qc_descriptor.u(
                description[0],
                description[1],
                0,
                j
            )
            qc=qc_descriptor+self._circuit
            
            t_qc = transpile(qc,
                            self.backend)
            qobj = assemble(t_qc,
                            parameter_binds = [{self.ksi: ksi} for ksi in list_ksi])
            job = self.backend.run(qobj)
            result = job.result()
            outputstate = result.get_statevector(qc, decimals=100)
            o = outputstate

            energy += np.real(o[0])
        
        
        return energy/n_atoms
    
    
    
class HybridFunction(Function):
    """ Hybrid quantum - classical function definition """
    
    @staticmethod
    def forward(ctx, input, quantum_circuit, shift,idx):
        """ Forward pass computation """
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit

        expectation_z = ctx.quantum_circuit.run(input[0].tolist(),idx)
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(input, result)

        return result
        
    @staticmethod
    def backward(ctx, grad_output,idx):
        """ Backward pass computation """
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())
        
        shift_right = input_list + np.ones(input_list.shape) * ctx.shift
        shift_left = input_list - np.ones(input_list.shape) * ctx.shift
        
        gradients = []
        for i in range(len(input_list)):
            expectation_right = ctx.quantum_circuit.run(shift_right[i],idx)
            expectation_left  = ctx.quantum_circuit.run(shift_left[i],idx)
            
            gradient = torch.tensor([expectation_right]) - torch.tensor([expectation_left])
            gradients.append(gradient)
        gradients = np.array([gradients]).T
        return torch.tensor([gradients]).float() * grad_output.float(), None, None

class Hybrid(nn.Module):
    """ Hybrid quantum - classical layer definition """
    
    def __init__(self, n_qubits, depth, backend, shift):
        super(Hybrid, self).__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, depth, backend)
        self.shift = shift
        
        
    def forward(self, input,idx):
        return HybridFunction.apply(input, self.quantum_circuit, self.shift,idx)
    


class Net(nn.Module):
    def __init__(self,n_qubits, depth):
        super(Net, self).__init__()
        self.linear=nn.Linear(1,n_qubits*(2*depth+3))
        self.hybrid=Hybrid(n_qubits,depth,backend=Aer.get_backend('aer_simulator'),shift=np.pi / 2)
        
    def forward(self,x,idx):
        x=self.linear(1)
        x=Hybrid(x,idx)
        

model=Net(6,2)
optimizer = optim.Adam(model.parameters(), lr=0.001)