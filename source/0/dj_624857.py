# https://github.com/jcking1034/qiskit-algorithms/blob/b405ff8b6e63bd55d3e33cd8d0c378ba42816d2c/dj.py
"Deutsch-Jozsa"
import enum
import sys

import numpy as np
import qiskit
from qiskit import IBMQ
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.quantum_info.operators import Operator

USE_IBMQ = True
SIMULATOR_NOISE = True
VERBOSE = True
NUM_SHOTS = 1024

if USE_IBMQ or SIMULATOR_NOISE:
    IBMQ.save_account("YOUR_KEY_HERE")


class Outcome(enum.Enum):
    CONSTANT = enum.auto()
    BALANCED = enum.auto()


def get_U_f_matrix(n_inp, n_anc, f):
    side = 2 ** (n_inp + n_anc)
    ret = np.zeros((side, side))

    for i in range(side):
        ret[bool(f(i >> n_anc)) ^ i, i] = 1

    return ret


def create_dj_circuit(n, f):
    circuit = qiskit.QuantumCircuit(n + 1, n)

    # Initialize last qubit to 1
    circuit.x(n)

    circuit.barrier()

    for i in range(n + 1):
        circuit.h(i)

    circuit.barrier()

    U_f = Operator(get_U_f_matrix(n, 1, f))
    circuit.unitary(U_f, reversed(range(n + 1)), "U_f")

    circuit.barrier()

    for i in range(n):
        circuit.h(i)

    circuit.measure(range(n), range(n))

    return circuit


if __name__ == "__main__":
    # functions = [
    #     lambda x: x & 1 == 0,
    #     lambda x: x & 1 == 1,
    #     lambda _: True,
    #     lambda _: False,
    # ]
    functions = {
        "balanced": lambda x: x & 1 == 1,
        "constant": lambda _: True,
    }
    try:
        n = int(sys.argv[1])
        f = functions[sys.argv[2]]
        print(f"{n} qubits, {sys.argv[2]} function")
    except Exception:
        print("Usage: python dj.py [n] [type: constant|balanced]")
        exit(1)

    circuit = create_dj_circuit(n, f)

    # print("Circuit:")
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
        print("Circuit Results:", result_sim)

    counts = result_sim.get_counts()
    result = int(max(counts.keys(), key=lambda x: counts[x]), base=2)
    if result == 0:
        print("\tResult: Constant")
    else:
        print("\tResult: Balanced")
