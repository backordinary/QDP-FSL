# https://github.com/adityabadhiye/quantum-grover-boolean-sat/blob/cb3217367dea947a00332f9e82a38eeba24bbb99/phase_estimation.py
from qiskit import QuantumCircuit, Aer, execute
from math import pi, sin
from qiskit.compiler import transpile, assemble
from grover_operator import GroverOperator


def qft(n):  # returns circuit for inverse quantum fourier transformation for given n
    circuit = QuantumCircuit(n)

    def swap_registers(circuit, n):
        for qubit in range(n // 2):
            circuit.swap(qubit, n - qubit - 1)
        return circuit

    def qft_rotations(circuit, n):
        if n == 0:
            return circuit
        n -= 1
        circuit.h(n)
        for qubit in range(n):
            circuit.cp(pi / 2 ** (n - qubit), qubit, n)
        qft_rotations(circuit, n)

    qft_rotations(circuit, n)
    swap_registers(circuit, n)
    return circuit


class PhaseEstimation:
    def get_control_gft(self, label="QFT †"):
        qft_dagger = self.qft_circuit.to_gate().inverse()
        qft_dagger.label = label
        return qft_dagger

    def get_main_circuit(self):
        qc = QuantumCircuit(self.c_bits + self.s_bits, self.c_bits)  # Circuit with n+t qubits and t classical bits

        # Initialize all qubits to |+>
        qc.h(range(self.c_bits + self.n_bits))
        qc.h(self.c_bits + self.s_bits - 1)
        qc.z(self.c_bits + self.s_bits - 1)
        # Begin controlled Grover iterations
        iterations = 1
        for qubit in range(self.c_bits):
            for i in range(iterations):
                qc.append(self.c_grover, [qubit] + [*range(self.c_bits, self.s_bits + self.c_bits)])
            iterations *= 2

        # Do inverse QFT on counting qubits
        qc.append(self.c_qft, range(self.c_bits))

        # Measure counting qubits
        qc.measure(range(self.c_bits), range(self.c_bits))
        return qc

    def simulate(self):
        aer_sim = Aer.get_backend('aer_simulator')
        transpiled_qc = transpile(self.main_circuit, aer_sim)
        qobj = assemble(transpiled_qc)
        job = aer_sim.run(qobj)
        hist = job.result().get_counts()
        # plot_histogram(hist)
        measured_int = int(max(hist, key=hist.get), 2)
        theta = (measured_int / (2 ** self.c_bits)) * pi * 2
        N = 2 ** self.n_bits
        M = N * (sin(theta / 2) ** 2)
        # print(N - M, round(N - M))
        return round(N - M)

    def __init__(self, grover: GroverOperator, c_bits=5):
        self.c_grover = grover.get_control_circuit()
        self.c_bits = c_bits
        self.n_bits = grover.n_bits
        self.s_bits = grover.total_bits
        self.qft_circuit = qft(c_bits)
        self.c_qft = self.get_control_gft()
        self.main_circuit = self.get_main_circuit()
        self.M = self.simulate()
