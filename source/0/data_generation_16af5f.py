# https://github.com/ironmanaudi/Machine-Learning-for-QEM/blob/85c35e59ae8480d67f8e631c89f5ce8cb7091a7e/data_generation.py
from qiskit import *
import numpy as np
from qiskit import execute, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Kraus, SuperOp
from qiskit.providers.aer import QasmSimulator
from qiskit.tools.visualization import plot_histogram

# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import QuantumError, ReadoutError
from qiskit.providers.aer.noise import pauli_error
from qiskit.providers.aer.noise import depolarizing_error
import qiskit.providers.aer.noise as noise
from qiskit.providers.aer.noise import thermal_relaxation_error
from qiskit import QuantumCircuit, execute
from qiskit import IBMQ, Aer
from qiskit.visualization import plot_histogram
from qiskit.providers.aer.noise import NoiseModel
from qiskit.circuit.random import random_circuit
import torch
from operator import itemgetter
# display(plot_histogram(result_ideal.get_counts()))


def generate_data(size, shots, num_qubits, depth, max_operands, noise_model, basis_gates):
    # data_ideal = torch.zeros((size, num_qubits, 2), dtype=torch.float64)
    # data_noisy = torch.zeros((size, num_qubits, 2), dtype=torch.float64)
    val_ideals = []
    val_noisys = []
    idxs = []
    sizes = []

    backend = Aer.get_backend('qasm_simulator')

    for i in range(size):
        circ = random_circuit(num_qubits, depth, max_operands, measure=True)

        result_ideal = execute(circ, backend ,basis_gates=basis_gates, shots=shots).result()
        result_ideal = result_ideal.get_counts()
        result_noisy = execute(circ,backend,basis_gates=basis_gates,noise_model=noise_model,shots=shots).result()
        result_noisy = result_noisy.get_counts()

        # for k in range(len(items_ideal)):
        #     for j in range(num_qubits):
        #         if items_ideal[k][0][j] == '0': data_ideal[i,j,0] += items_ideal[k][1]
        #         else: data_ideal[i,j,1] += items_ideal[k][1]
        # for l in range(len(items_noisy)):
        #     for m in range(num_qubits):
        #         if items_noisy[l][0][m] == '0': data_noisy[i,m,0] += items_noisy[l][1]
        #         else: data_noisy[i,m,1] += items_noisy[l][1]
        
        #update dict of ideal and noisy
        temp_ideal = dict.fromkeys(result_ideal, 0)
        temp_noisy = dict.fromkeys(result_noisy, 0)
        result_ideal.update(temp_noisy)
        result_noisy.update(temp_ideal)
        idx = sorted(result_ideal)
        val_ideal = list(itemgetter(*idx)(result_ideal))
        val_noisy = list(itemgetter(*idx)(result_noisy))
        #append noisy prob for each index
        for k in range(len(idx)):
            idx[k] = list(idx[k])
            idx[k].append(val_noisy[k]/shots)
            
        sizes.append(len(idx))
                
        #convert idx from str to float
        idx = np.array(idx)
        idx = idx.astype(np.float64)
        idx = torch.from_numpy(idx)
        
        #list of tensors
        val_ideals.append(torch.Tensor(val_ideal, dtype=torch.float64)/shots)
        val_noisys.append(torch.Tensor(val_noisy, dtype=torch.float64)/shots)
        idxs.append(idx)
    
    #pad noisy and ideal
    idxs = torch.nn.utils.rnn.pad_sequence(idxs, batch_first=True)
    val_ideals = torch.nn.utils.rnn.pad_sequence(val_ideals, batch_first=True)
    val_noisys = torch.nn.utils.rnn.pad_sequence(val_noisys, batch_first=True)
    
    #pack padded sequences
    pack_idxs = torch.nn.utils.rnn.pack_padded_sequence(idxs, lengths=sizes, batch_first=True, enforce_sorted=False)
    pack_ideals = torch.nn.utils.rnn.pack_padded_sequence(val_ideals, lengths=sizes, batch_first=True, enforce_sorted=False)
    pack_noisys = torch.nn.utils.rnn.pack_padded_sequence(val_noisys, lengths=sizes, batch_first=True, enforce_sorted=False)
    
    return (val_ideals, pack_idxs, sizes)


def noisy_model(prob_one, prob_two):
    # Depolarizing quantum errors
    error_1 = noise.depolarizing_error(prob_one, 1)
    error_2 = noise.depolarizing_error(prob_two, 2)

    # Add errors to noise model
    noise_model = noise.NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
    basis_gates = noise_model.basis_gates

    return (noise_model, basis_gates)


if __name__ == "__main__":
    size = 2
    shots = 2048
    num_qubits = 4
    depth = 3
    max_operands = 2
    prob_one =0.01
    prob_two = 0.01
    
    noise_model, basis_gates = noisy_model(prob_one, prob_two)
    data_ideal, data_noisy = generate_data(size, shots, num_qubits, depth, max_operands, noise_model, basis_gates)
    print('1', data_ideal)
    print('2', data_noisy)

