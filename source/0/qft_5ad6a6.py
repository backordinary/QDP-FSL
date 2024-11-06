# https://github.com/ogorodnikov/m1/blob/dc9364b445aea1fff137c294ef01981b148d70f4/app/core-service/core/algorithms/qft.py
from math import pi
from itertools import combinations_with_replacement

from qiskit import QuantumCircuit
from qiskit import Aer


def create_qft_circuit(qubits_count,
                       flipped=False,
                       inverted=False,
                       barriers=True):
    
    if not flipped:
        
        qubits = range(qubits_count)
        
    else:
        
        qubits = reversed(range(qubits_count))
    
    qubit_pairs = combinations_with_replacement(qubits, 2)
    
    circuit = QuantumCircuit(qubits_count)
    
    if not inverted:
        
        pairs = qubit_pairs
        circuit.name = 'QFT'
        
    else:
    
        pairs = reversed(tuple(qubit_pairs))
        circuit.name = 'QFT†'

    for control_qubit, target_qubit in pairs:
        
        if control_qubit == target_qubit:
            
            if barriers:
                circuit.barrier()
                
            circuit.h(control_qubit)
            
        else:
            
            distance = abs(control_qubit - target_qubit)
            
            if not inverted:
                theta = pi / 2 ** distance
            else:
                theta = -pi / 2 ** distance
                
            circuit.cp(theta, control_qubit, target_qubit)
            
        #     task_log(f'QFT distance: {distance}')
        #     task_log(f'QFT theta: {theta}')
            
        # task_log(f'QFT control_qubit: {control_qubit}')
        # task_log(f'QFT target_qubit: {target_qubit}')
        # task_log(f'QFT circuit: \n{circuit}')
    
    return circuit


def qft(run_values, task_log):
    
    input_number = run_values.get('number')

    number = ''.join('1' if digit == '1' else '0' for digit in input_number)
    
    qubits_count = len(number)
    qubits = range(qubits_count)
    
    measure_bits_count = qubits_count
    measure_bits = range(measure_bits_count)

    circuit = QuantumCircuit(qubits_count, measure_bits_count)
    circuit.name = 'QFT Circuit'
    
    for input_qubit, digit in enumerate(number):
        
        if digit == '1':
            circuit.x(input_qubit)
            

    qft_circuit = create_qft_circuit(qubits_count)
    
    circuit.append(qft_circuit, qubits)
    
    iqft_circuit = qft_circuit.inverse()
    iqft_circuit.name = 'IQFT Circuit'

    task_log(f'QFT input_number: {input_number}')
    task_log(f'QFT number: {number}')
    task_log(f'QFT qubits_count: {qubits_count}')

    task_log(f'QFT qft_circuit: \n{qft_circuit}')
    task_log(f'QFT iqft_circuit: \n{iqft_circuit}')
    
    task_log(f'QFT circuit: \n{circuit}')

    return circuit