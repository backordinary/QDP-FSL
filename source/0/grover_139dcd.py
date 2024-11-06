# https://github.com/jcking1034/qiskit-algorithms/blob/b405ff8b6e63bd55d3e33cd8d0c378ba42816d2c/grover.py
"""Grover"""
import random
import sys
import time
from math import acos, floor, pi, sin, sqrt

import numpy as np
import qiskit
from qiskit import IBMQ
from qiskit.circuit.library.standard_gates import XGate
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.quantum_info.operators import Operator

USE_IBMQ = True
SIMULATOR_NOISE = False
VERBOSE = True
NUM_SHOTS = 1024

if USE_IBMQ or SIMULATOR_NOISE:
    IBMQ.save_account("YOUR_KEY_HERE")

# poetry run python grover.py 8 10


def get_U_f_matrix(n_inp, n_anc, f):
    side = 2 ** (n_inp + n_anc)
    ret = np.zeros((side, side))

    for i in range(side):
        ret[bool(f(i >> n_anc)) ^ i, i] = 1

    return ret


def create_grover_circuit(n, f, a):
    k, _ = calc_grover_params(a, n)

    circuit = qiskit.QuantumCircuit(n + 1, n)

    circuit.x(n)

    circuit.barrier()

    for i in range(n + 1):
        circuit.h(i)

    circuit.barrier()

    for i in range(k):
        U_f = Operator(get_U_f_matrix(n, 1, f))
        circuit.unitary(U_f, reversed(range(n + 1)), "U_f")

        for j in range(n):
            circuit.h(j)

        mcx_gate = XGate().control(n, ctrl_state=0)
        circuit.append(mcx_gate, range(n + 1))

        for j in range(n):
            circuit.h(j)

    for i in range(n):
        circuit.measure(i, n - i - 1)

    return circuit


def calc_grover_params(a, n):
    N = 1 << n
    # the angle of the grover circuit as a rotation matrix in the special subspace
    theta = acos(1 - 2 * a / N) / 2
    # we choose the integer value of k which minimizes |pi/2 - (2k+1)*theta|
    # which is round(pi/(4*theta) - 1/2)
    # then note round(x) = floor(x+0.5)
    k = floor(pi / (4 * theta))
    final_sin = sin((2 * k + 1) * theta)
    success_prob = final_sin**2
    return k, success_prob


def run(n, f, a):
    """
    Creates and runs the Grover circuit for the given fixed value of a
    Input: f is a function that maps n-bit integers to {0,1}
    Returns a single integer: a value x such that f(x)=1 with good probability
    """
    circuit = create_grover_circuit(n, f, a)
    # print("Circuit")
    # print(circuit.draw())

    if USE_IBMQ:
        provider = IBMQ.load_account()
        backend = provider.backend.ibmq_quito
    elif SIMULATOR_NOISE:
        provider = IBMQ.load_account()
        noise_backend = provider.backend.ibmq_quito
        noise_model = NoiseModel.from_backend(noise_backend)
        backend = QasmSimulator(noise_model=noise_model)
    else:
        backend = QasmSimulator()

    # Execute the circuit on the qasm simulator.
    # We've set the number of repeats of the circuit
    # to be 1024, which is the default.
    transpiled = qiskit.transpile(circuit, backend)
    job_sim = backend.run(transpiled, shots=NUM_SHOTS)

    # Grab the results from the job.
    result_sim = job_sim.result()
    if VERBOSE:
        print(result_sim)

    counts = result_sim.get_counts()
    # print(counts)

    result_int = int(max(counts.keys(), key=lambda x: counts[x]), base=2)

    # Qubit q{i} represents the ith *most* significant bit of the input to f
    # when we are constructing U_f
    # Here we account for that by shifting q{i} by (n-i-1) in the output
    # The caller of run_grover does not need to think about this, as they receive
    # an integer x which they can plug into f (and hopefully get f(x)=1)
    return (result_int, result_sim.time_taken)


def main_extended(max_num_bits, num_cases):
    for num_bits in range(1, max_num_bits + 1):
        start_time_inc_setup = time.perf_counter()
        total_run_time = 0
        num_correct = 0
        for i in range(num_cases):
            N = 1 << num_bits
            true_a = random.randint(1, N / 2)
            r = np.random.permutation([0] * true_a + [1] * (N - true_a))

            def f(x):
                return r[x]

            overall_correct = 0
            for j in range(num_bits):
                a = floor(sqrt(1 << j))
                result, run_time = run(num_bits, f, a)
                overall_correct = overall_correct or f(result)
                total_run_time += run_time
            num_correct += overall_correct
        end_time_inc_setup = time.perf_counter()
        print("")
        print(f"Grover (unknown number of 1s) for n={num_bits} bits")
        print(f"Total correct: {num_correct}/{num_cases}")
        print(
            f"Total run time including setup: {end_time_inc_setup-start_time_inc_setup:3f} s"
        )
        print(f"Average simulation run time: {total_run_time/num_cases:3f} s")


def main(max_num_bits, num_cases):
    for num_bits in range(1, max_num_bits + 1):
        start_time_inc_setup = time.perf_counter()
        total_run_time = 0
        num_correct = 0
        for i in range(num_cases):
            N = 1 << num_bits
            # randint(a,b) samples from [a,b] inclusive
            r = random.randint(0, N - 1)
            print("r: ", r)

            def f(x):
                return int(x == r)

            result, run_time = run(num_bits, f, 1)
            num_correct += f(result)
            total_run_time += run_time
        end_time_inc_setup = time.perf_counter()
        print("")
        k, success_prob = calc_grover_params(1, num_bits)
        print(f"Grover (1-hot) for n={num_bits} bits")
        print(f"Using k={k} iterations")
        print(f"Expected correct: {floor(success_prob*num_cases+0.5)}/{num_cases}")
        print(f"Total correct: {num_correct}/{num_cases}")
        print(
            f"Total run time including setup: {end_time_inc_setup-start_time_inc_setup:3f} s"
        )
        print(f"Average simulation run time: {total_run_time/num_cases:3f} s")


if __name__ == "__main__":
    max_num_bits = None
    num_cases = None
    if len(sys.argv) == 3:
        max_num_bits, num_cases = tuple(map(int, sys.argv[1:]))
        main(max_num_bits, num_cases)
    elif len(sys.argv) == 4 and sys.argv[1] == "--extended":
        max_num_bits, num_cases = tuple(map(int, sys.argv[2:]))
        main_extended(max_num_bits, num_cases)
    else:
        print("Usage: ./grover.py [--extended] MAX_NUM_BITS NUM_CASES")
        exit(1)
