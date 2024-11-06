# https://github.com/Komoszek/QuantumHashing/blob/c6fb17b9ecf3f9b3bdb6cf30ce509e0cdaa69a06/quantumHashing.py
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import RYGate
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import math
from qiskit import Aer, execute

class QuantumHash:
    def __init__(self, backend=Aer.get_backend('qasm_simulator'), msg=[], qubits_num=0):
        self.backend = backend
        self._msg = msg
        self._qubits_num = qubits_num
        self._circuit = QuantumCircuit(self.qubits_num, name='Quantum Hasher')
        self._k = []
        self._constructed_circuit = True
        self._result = None

    def get_result(self, recalculate=False):
        if recalculate or not self._constructed_circuit or self._result is None:
            self.recalculate_result()
        return self._result

    def recalculate_result(self, shots=1000):
        if not self._constructed_circuit:
            self.generate_circuit()
        self._result = execute(self._circuit.measure_all(inplace=False), self.backend, shots=shots).result()

    def generate_circuit(self):
        self._circuit = QuantumCircuit(self.qubits_num)
        control_qubits_num = self.control_qubits_num
        N = 2 ** len(self.msg)
        K = 2 ** control_qubits_num
        bit = 1
        self._circuit.h(range(0, control_qubits_num))

        for i in range(len(self.msg)):
            if self.msg[i]:
                for j in range(K):
                    gate = RYGate((4 * math.pi * bit * self.k[j]) / N).control(control_qubits_num, ctrl_state=j)
                    self._circuit.append(gate, range(self.qubits_num))

            bit *= 2
        self.result = None
        self._constructed_circuit = True
        self._measure_circuit = self._circuit.measure_all(inplace=False)

    def show_circuit(self, output='mpl'):
        if not self._constructed_circuit:
            self.generate_circuit()
        self._circuit.draw(output=output)
        plt.show()

    def show_histogram(self):
        res = self.get_result().get_counts(self._circuit.measure_all(inplace=False))
        plot_histogram(res, title='Hash histogram')
        plt.show()

    @property
    def msg(self):
        return self._msg

    @msg.setter
    def msg(self, value):
        self._msg = value

    @property
    def circuit(self):
        if not self._constructed_circuit:
            self.generate_circuit()
        return self._circuit

    @property
    def control_qubits_num(self):
        return self._qubits_num - 1

    @property
    def qubits_num(self):
        return self._qubits_num

    @control_qubits_num.setter
    def control_qubits_num(self, value):
        self.qubits_num = value + 1

    @qubits_num.setter
    def qubits_num(self, value):
        if self._qubits_num != value:
            self._qubits_num = value
            self.k = [i + 1 for i in
                    range(2 ** self.control_qubits_num)] .reverse()
            self._constructed_circuit = False

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, value):
        self._k = value
        self._constructed_circuit = False

    def set_single_k(self, new_k_val, i):
        self.k[i] = new_k_val
        self._constructed_circuit = False