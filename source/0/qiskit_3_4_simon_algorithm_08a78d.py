# https://github.com/physicalattraction/quantum-computing-tutorial/blob/744ff30bd97a54a89241960af42c57b3c63a7506/src/qiskit_3_4_simon_algorithm.py
"""
https://qiskit.org/textbook/ch-algorithms/simon.html
"""

import random

import numpy as np
from itertools import product
from qiskit import QuantumCircuit
from qiskit_textbook.tools import simon_oracle

from utils import draw_quantum_circuit, print_histogram_from_real

BitString = str


def get_secret_b(n: int) -> BitString:
    """
    Return a random bit string of length n, e.g. 10001100
    """

    return ''.join([str(np.random.randint(2)) for _ in range(n)])


def xor(bit_string_1: str, bit_string_2: str) -> str:
    """
    Return the bitwise-XOR value of the two bit strings

    >>> xor('110', '101')
    '011'
    >>> xor('110', '011')
    '101'
    """

    result = ''
    for bit_1, bit_2 in zip(bit_string_1, bit_string_2):
        result += str((int(bit_1) + int(bit_2)) % 2)
    return result


def f(input_string: BitString, secret_string: BitString) -> BitString:
    """
    Apply the function f with the secret bit string s

    In words, f counts the number of bits that are 1 in both the input string and the secret string,
    and returns that count modulo 2
    """

    if '1' not in secret_string:
        return input_string

    xor_bit_string = xor(input_string, secret_string)
    return min(input_string, xor_bit_string)


def test_f():
    inputs = [''.join(p) for p in product('01', repeat=3)]

    outputs = {b: f(b, '000') for b in inputs}
    assert outputs == {'000': '000', '001': '001', '010': '010', '011': '011',
                       '100': '100', '101': '101', '110': '110', '111': '111'}
    assert len(set(outputs.values())) == 8

    outputs = {b: f(b, '001') for b in inputs}
    assert outputs == {'000': '000', '001': '000', '010': '010', '011': '010',
                       '100': '100', '101': '100', '110': '110', '111': '110'}
    assert len(set(outputs.values())) == 4

    inputs = [''.join(p) for p in product('01', repeat=8)]
    outputs = {b: f(b, '00000000') for b in inputs}
    assert len(set(outputs.values())) == 256, len(set(outputs.values()))
    outputs = {b: f(b, '00100101') for b in inputs}
    assert len(set(outputs.values())) == 128, len(set(outputs.values()))
    random_value = random.choice(list(outputs.values()))
    keys_for_that_value = [key for key, value in outputs.items() if value == random_value]
    assert len(keys_for_that_value) == 2
    assert xor(*keys_for_that_value) == '00100101'

    print('All tests passed')


def classical_solution(b: str, verbose: bool = False):
    result = {}  # Dict from output of f to input of f
    inputs = [''.join(p) for p in product('01', repeat=len(b))]
    random.shuffle(inputs)
    for index, input in enumerate(inputs):
        if index > 2 ** (len(b) - 1):
            guess = '0' * len(b)
            if verbose:
                print(f'Found {guess} in {index} function calls')
            return index
        output = f(input, b)
        if output in result:
            guess = xor(result[output], input)
            if verbose:
                print(f'Found {guess} in {index} function calls')
            return index
        else:
            result[output] = input
    raise NotImplementedError(f'ERROR! Secret string {b} cannot be found')


def analyze_classical():
    """
    Analyze the number of required function calls for increasing bit string lengths

    Example for up to n=19:
    {2: 1.96, 3: 3.06, 4: 4.52, 5: 7.03, 6: 8.69, 7: 12.48, 8: 20.53, 9: 28.24, 10: 36.88, 11: 53.04,
     12: 80.68, 13: 106.59, 14: 170.37, 15: 232.34, 16: 281.72, 17: 458.03, 18: 644.18, 19: 935.31}
    """

    result = {}
    total_count = 0
    for _ in range(100):
        b = get_secret_b(2)
        total_count += classical_solution(b)
    result[2] = total_count / 100
    print(result)
    total_count = 0
    for _ in range(100):
        b = get_secret_b(3)
        total_count += classical_solution(b)
    result[3] = total_count / 100
    print(result)
    total_count = 0
    for _ in range(100):
        b = get_secret_b(4)
        total_count += classical_solution(b)
    result[4] = total_count / 100
    print(result)
    total_count = 0
    for _ in range(100):
        b = get_secret_b(5)
        total_count += classical_solution(b)
    result[5] = total_count / 100
    print(result)
    total_count = 0
    for _ in range(100):
        b = get_secret_b(6)
        total_count += classical_solution(b)
    result[6] = total_count / 100
    print(result)
    total_count = 0
    for _ in range(100):
        b = get_secret_b(7)
        total_count += classical_solution(b)
    result[7] = total_count / 100
    print(result)
    total_count = 0
    for _ in range(100):
        b = get_secret_b(8)
        total_count += classical_solution(b)
    result[8] = total_count / 100
    print(result)
    total_count = 0
    for _ in range(100):
        b = get_secret_b(9)
        total_count += classical_solution(b)
    result[9] = total_count / 100
    print(result)
    total_count = 0
    for _ in range(100):
        b = get_secret_b(10)
        total_count += classical_solution(b)
    result[10] = total_count / 100
    print(result)


def get_simon_circuit(b: str):
    n = len(b)
    simon_circuit = QuantumCircuit(n * 2, n)

    # Apply Hadamard gates before querying the oracle
    simon_circuit.h(range(n))

    # Apply barrier for visual separation
    simon_circuit.barrier()

    # Apply the oracle function
    simon_circuit = simon_circuit.compose(simon_oracle(b))

    # Apply barrier for visual separation
    simon_circuit.barrier()

    # Apply Hadamard gates to the input register
    simon_circuit.h(range(n))

    # Measure qubits
    simon_circuit.measure(range(n), range(n))

    return simon_circuit


def quantum_solution_simulated(b: str):
    qc = get_simon_circuit(b)
    counts = draw_quantum_circuit(qc, draw_final_state=False, draw_simulate=True)

    # Calculate the dot product of the results
    def bdotz(b, z):
        accum = 0
        for i in range(len(b)):
            accum += int(b[i]) * int(z[i])
        return accum % 2

    for z in counts:
        # Since we know b already, we can verify these results do satisfy b⋅z=0 (mod 2):
        print(f'{b}.{z} = {bdotz(b, z)} (mod 2)')
    print(f'There are {len(counts)} outputs for the possible {2 ** len(b)} inputs')

    # Using these results, we can recover the value of b=110 by solving this set of simultaneous equations,
    # with Gaussian elimination, which has a run time of O(n3): https://mathworld.wolfram.com/GaussianElimination.html
    # This is not implemented in this file, since it's not directly related to quantum computing.


def quantum_solution_real(b: str):
    qc = get_simon_circuit(b)
    print_histogram_from_real(qc)


if __name__ == '__main__':
    # b = get_secret_b(12)
    # classical_solution(b)
    # analyze_classical()
    secret_b = get_secret_b(2)
    quantum_solution_simulated(secret_b)
    quantum_solution_real(secret_b)
