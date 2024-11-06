# https://github.com/TateStaples/Quantum/blob/62e49dd97cee55d60f9b16494a73e5532a7f4494/framework.py
from qiskit import *
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
import matplotlib.pyplot as plt
from math import pi

simulator = None
max_qubits = 5
shot = 500
_circuit = None


def build_circuit(qubits=max_qubits, normal_bit=1):
    global _circuit
    _circuit = QuantumCircuit(qubits, normal_bit)
    Qubit.available_qubits = [True for i in range(qubits)]


def assert_circuit():
    if _circuit is None:
        build_circuit()



class Qubit:
    available_qubits = []

    def __init__(self):
        assert_circuit()
        self._establish()

    def _establish(self):  # have a way to get rid
        if any(self.available_qubits):
            self._index = self.available_qubits.index(True)
            Qubit.available_qubits[self._index] = False
        else:
            print(Qubit.available_qubits)
            raise AssertionError("ran out of qubits")

    @staticmethod
    def get_circuit():
        return _circuit

    def __and__(self, other):
        result = Qubit()
        _circuit.ccx(self._index, other._index, result._index)
        _circuit.x(result._index)
        return result

    def __or__(self, other):
        result = Qubit()
        _circuit.cx(self._index, result._index)
        _circuit.cx(other._index, result._index)
        _circuit.x(result._index)
        return result

    def __invert__(self):
        q = Qubit()
        _circuit.cx(self._index, q._index)
        return q

    def __del__(self):
        Qubit.available_qubits[self._index] = True

    def swap(self, other):
        self._index, other._index = other._index, self._index

    def copy(self):
        q = Qubit()
        _circuit.cx(self._index, q._index)
        _circuit.x(q._index)
        return q

    def reset(self):
        _circuit.reset(self._index)

    def superpose(self):
        _circuit.h(self._index)

    def cx(self, target):
        _circuit.cx(self._index, target._index)

    def x(self):
        _circuit.x(self._index)

    def root(self):
        _circuit.sx(self._index)

    def rotate(self, ø=pi/4, direction='rz'):
        if direction == 'rz': _circuit.rz(ø, self._index)
        elif direction == 'rx': _circuit.rx(ø, self._index)
        elif direction == 'ry': _circuit.ry(ø, self._index)

    @staticmethod
    def measure(qubit_indices, bit_indices):
        _circuit.measure(qubit_indices, bit_indices)

    @staticmethod
    def draw(title=''):
        state = Statevector.from_instruction(_circuit)
        plot_bloch_multivector(state, title=title)
        plt.show()


class QInt:
    def __init__(self):
        pass

    def __add__(self, other):
        pass


def get_result():
    global simulator, shot, _circuit
    if simulator is None:
        simulator = Aer.get_backend("qasm_simulator")
    return execute(_circuit, simulator, shots=shot).result()
