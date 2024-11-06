# https://github.com/ogorodnikov/m1/blob/c9a851b4eb244dd59ae6f2d10670b8742b39b5af/app/core-service/core/algorithms/bernvaz.py
from qiskit import QuantumCircuit


def bernvaz(run_values, task_log):
    
    secret = run_values.get('secret')
    
    task_id = run_values.get('task_id')

    hidden_qubit_count = len(secret)
    total_qubit_count = hidden_qubit_count + 1
    
    input_qubit_indices = range(hidden_qubit_count)
    measure_bits_indices = input_qubit_indices
    
    output_qubit_index = hidden_qubit_count
    
    all_qubit_indices = range(total_qubit_count)
    
    
    circuit = QuantumCircuit(total_qubit_count, hidden_qubit_count)
    
    circuit.x(output_qubit_index)
    circuit.h(all_qubit_indices)
    
    circuit.barrier()
    
    for input_qubit_index, digit in enumerate(reversed(secret)):
        if digit == '1':
            circuit.cx(input_qubit_index, output_qubit_index)
            
    circuit.barrier()
            
    circuit.h(input_qubit_indices)
    
    circuit.measure(input_qubit_indices, measure_bits_indices)
    
    task_log(f"BERVAZ run_values: {run_values}")
    task_log(f"BERVAZ secret: {secret}")
    task_log(f"BERVAZ circuit: \n{circuit}\n")
    
    return circuit