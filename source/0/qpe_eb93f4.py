# https://github.com/ogorodnikov/m1/blob/9f23da1406a679eaae545a1721b11e06310ca58b/app/core-service/core/algorithms/qpe.py
from math import pi

from qiskit import QuantumCircuit, Aer

from core.algorithms.qft import create_qft_circuit


def qpe(run_values, task_log):
    
    angle = run_values.get('angle')
    precision = run_values.get('precision')
    
    angle_number = float(angle)

    counting_qubits_count = int(precision)
    counting_qubits = range(counting_qubits_count)
    
    eigenstate_qubit = max(counting_qubits) + 1
    qubits_count = counting_qubits_count + 1
    
    measure_bits_count = counting_qubits_count
    measure_bits = range(measure_bits_count)
    
    qubits_measurement_list = list(reversed(counting_qubits))

    circuit = QuantumCircuit(qubits_count, measure_bits_count)
    circuit.name = 'QPE Circuit'
    
    for counting_qubit in counting_qubits:
        circuit.h(counting_qubit)
        
    circuit.x(eigenstate_qubit)
    
    
    for repetitions, counting_qubit in enumerate(counting_qubits):
        
        for i in range(2 ** repetitions):
        
            circuit.cp(pi * angle_number, counting_qubit, eigenstate_qubit)
            
            
    qft_dagger_circuit = create_qft_circuit(counting_qubits_count, inverted=True)
    
    circuit.append(qft_dagger_circuit, counting_qubits)
    
    circuit.barrier()
    
    circuit.measure(qubits_measurement_list, measure_bits)
    
    
    task_log(f'QPE angle: \n{angle}')
    task_log(f'QPE precision: \n{precision}')
    
    task_log(f'QPE qft_dagger_circuit: \n{qft_dagger_circuit}')
    task_log(f'QPE circuit: \n{circuit}')

    return circuit
    
    
def qpe_post_processing(run_data, task_log):
    
    run_result = run_data.get('Result')
    counts = run_result.get('Counts')
    
    counts_decimals = {int(state, 2): count for state, count in counts.items()}
    
    counts_sum = sum(counts_decimals.values())
    
    states_weighted_sum = sum(state_decimal * count for state_decimal, count 
                              in counts_decimals.items())
    
    states_average = states_weighted_sum / counts_sum
    
    first_state = next(iter(counts))
    
    precision = len(first_state)
    
    theta = states_average / 2 ** precision
    
    angle = theta * 2

    task_log(f'QPE qpe_post_processing')
    
    task_log(f'QPE counts: {counts}')
    task_log(f'QPE counts_decimals: {counts_decimals}')
    task_log(f'QPE counts_sum: {counts_sum}')
    task_log(f'QPE states_weighted_sum: {states_weighted_sum}')
    task_log(f'QPE states_average: {states_average}')
    task_log(f'QPE first_state: {first_state}')
    
    task_log(f'QPE precision: {precision}')
    task_log(f'QPE theta: {theta}')
    task_log(f'QPE angle: {angle}')
    
    return {'Angle': angle}