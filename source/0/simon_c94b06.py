# https://github.com/ogorodnikov/m1/blob/d93902a70ba5b15f86d45b775f9cbac99f3dc95d/app/core-service/core/algorithms/simon.py
from random import randint
from itertools import combinations

from sympy import Matrix, symbols, mod_inverse, linsolve
from sympy.core.numbers import Zero

from qiskit import QuantumCircuit


def build_simon_oracle(period, task_log, masquerade=True):
    
    input_qubits_count = len(period)
    output_qubits_count = input_qubits_count
    all_qubits_count = input_qubits_count + output_qubits_count
    
    input_qubits = range(input_qubits_count)
    output_qubits = range(output_qubits_count, 2 * output_qubits_count)
    all_qubits = range(all_qubits_count)
    
    oracle = QuantumCircuit(all_qubits_count)
    oracle.name = 'Simon Oracle'
    
    
    for input_qubit, output_qubit in zip(input_qubits, output_qubits):
        
        oracle.cx(input_qubit, output_qubit)
        
        
    is_one_to_one_mapping = '1' not in period
    
    if is_one_to_one_mapping:
        
        return oracle
        
    
    first_period_one_bit = period.find('1')
    
    for output_qubit, digit in enumerate(period, input_qubits_count):
        
        if digit == '1':
            
            oracle.cx(first_period_one_bit, output_qubit)
    
            
    if masquerade:
        
        oracle.barrier()
        
        qubit_pairs = list(combinations(output_qubits, 2))
        
        for a, b in qubit_pairs:
            
            is_swapped = randint(0, 1)
            
            if is_swapped:
                
                oracle.swap(a, b)
                

        oracle.barrier()
        
        for output_qubit in output_qubits:
            
            is_flipped = randint(0, 1)
            
            if is_flipped:
                
                oracle.x(output_qubit)  
    
    
    return oracle
    

    

def simon(run_values, task_log):
        
    input_period = run_values.get('period')
    masquerade = run_values.get('masquerade')
    masquerade_bool = masquerade.lower() == 'true'
    
    period = ''.join('1' if digit == '1' else '0' for digit in input_period)
    
    input_qubits_count = len(period)
    
    output_qubits_count = input_qubits_count
    all_qubits_count = input_qubits_count + output_qubits_count
    
    input_qubits = range(input_qubits_count)
    all_qubits = range(all_qubits_count)
    
    measure_bits_count = input_qubits_count
    measure_bits = range(measure_bits_count)


    simon_oracle = build_simon_oracle(period, task_log, masquerade=masquerade_bool)
    

    circuit = QuantumCircuit(all_qubits_count, measure_bits_count)
    
    for input_qubit in input_qubits:    
        circuit.h(input_qubit)
        
    circuit.append(simon_oracle, all_qubits)
        
    for input_qubit in input_qubits:    
        circuit.h(input_qubit)
        

    qubits_measurement_list = list(reversed(input_qubits))
    
    circuit.measure(qubits_measurement_list, measure_bits)
    
    
    task_log(f'SIMON input_period: {input_period}')
    task_log(f'SIMON period: {period}')
    task_log(f'SIMON masquerade: {masquerade}')
    task_log(f'SIMON masquerade_bool: {masquerade_bool}')

    task_log(f'SIMON input_qubits_count: {input_qubits_count}')
    task_log(f'SIMON all_qubits_count: {all_qubits_count}')
    task_log(f'SIMON qubits_measurement_list: {qubits_measurement_list}')
    
    task_log(f'SIMON simon_oracle: \n{simon_oracle}')
    task_log(f'SIMON circuit: \n{circuit}')
    
    return circuit
    
    
def sympy_modulus(x, modulus):

    numerator, denominator = x.as_numer_denom()

    return numerator * mod_inverse(denominator, modulus) % modulus


def simon_post_processing(run_data, task_log):
    
    run_result = run_data.get('Result')
    counts = run_result.get('Counts')
    
    counts_median = max(counts.values()) / 2
    
    filtered_solutions = [solution for solution, count in counts.items()
                          if set(solution) != {'0'}
                          and count > counts_median]
                          
    digits = [list(map(int, solution)) for solution in filtered_solutions]
    
    
    modulus_two = lambda x: sympy_modulus(x, 2)
    
    is_even = lambda x: x % 2 == 0
    
    
    matrix = Matrix(digits)
    
    rref_matrix = matrix.rref(pivots=False, iszerofunc=is_even)
    
    rref_mod_matrix = rref_matrix.applyfunc(modulus_two)
    
    
    qubits_count = len(digits[0])
    
    variables = symbols(f'z:{qubits_count}')
    
    zeros = Matrix.zeros(rref_mod_matrix.rows, 1)
    
    system = rref_mod_matrix, zeros
    
    
    solutions = linsolve(system, *variables)
    
    first_solution = next(iter(solutions))
    
    period = ''.join('0' if isinstance(symbol, Zero) else '1'
                     for symbol in first_solution)
    

    task_log(f'SIMON simon_post_processing')
    task_log(f'SIMON counts: {counts}')
    task_log(f'SIMON counts_median: {counts_median}')
    task_log(f'SIMON filtered_solutions: {filtered_solutions}')

    task_log(f'SIMON matrix: ' + repr(matrix))
    task_log(f'SIMON rref_matrix: ' + repr(rref_matrix))
    task_log(f'SIMON rref_mod_matrix: ' + repr(rref_mod_matrix))
    
    task_log(f'SIMON qubits_count: {qubits_count}')
    task_log(f'SIMON variables: {variables}')

    task_log(f'SIMON solutions: {solutions}')  
    task_log(f'SIMON period: {period}')
    
    
    return {'Period': period}