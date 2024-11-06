# https://github.com/wsdt/QuantumComputing_Python/blob/e52b0c833dc146b855b2962b84cf3f22832b4c4e/src/quantum_true_randomstring_generator.py
# (c) Riedl A. Kevin, 2019
# https://github.com/wsdt
# https://www.linkedin.com/in/kevin-riedl-947219158

# Inspired from https://github.com/robbiemccorkell/quantum-random-number

from conf.general_conf import QISKIT_RAND_STR_LEN
import math
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, BasicAer
from conf.controllers.c_qiskit_conf import get_quantum_machine

# Local constant for maximum qubits to use
MAX_QUBITS = 5


def get_register_sizes(n):
    register_sizes = [MAX_QUBITS for _ in range(int(n/MAX_QUBITS))]
    remainder = n % MAX_QUBITS
    return register_sizes if remainder == 0 else register_sizes+[remainder]


# Get number of bits from number value
def num_bits(n):
    return math.floor(math.log(n, 2))+1


def bit_from_counts(counts):
    return [k for k, v in counts.items() if v == 1][0]


# Enforce min value to have a visible string (ascii table) as we just allocate
# memory which will then contain random bit values. To enforce min the value will be
# adapted recursively.
# @return Adapted random number.
def cut_no_down_to_min(max_bound, min_bound, rand_no):
    if rand_no < min_bound:
        tmp_min = min_bound-rand_no
        rand_no += cut_no_down_to_min(max_bound-rand_no, 0, random_int(tmp_min))

    return rand_no


# Generate random integer via qbits
# @return Returns truly random integer
def random_int(max_bound):
    bits = ''
    n_bits = num_bits(max_bound - 1)
    register_sizes = get_register_sizes(n_bits)
    backend = get_quantum_machine()

    for x in register_sizes:
        q = QuantumRegister(x)
        c = ClassicalRegister(x)
        qc = QuantumCircuit(q, c)

        # H-Gate, get into superposition
        qc.h(q)
        # Measure value of superposition
        qc.measure(q, c)

        # Execute quantum circuit on specific backend
        job_sim = execute(qc, backend, shots=1)
        sim_result = job_sim.result()
        counts = sim_result.get_counts(qc)

        bits += bit_from_counts(counts)
    return int(bits, 2)


# Parse/Cast generated integer into char value
def random_char(rand_int):
    return chr(rand_int)


# Generate random string with specific length.
# @return Generated String
def random_str(length=10):
    res_str = ""
    for _ in range(length):
        res_str += random_char(
            cut_no_down_to_min(125, 33, random_int(125))
        )

    return res_str


# Generate random string of specific length and print it afterwards
result = random_str(QISKIT_RAND_STR_LEN)
print(result)
