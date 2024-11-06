# https://github.com/Yan-Wang88/cmput-604-project/blob/7ba7e15f9d11c1b2a001bf4f3b150e151cdd0715/channel.py
from qiskit import Aer
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import matplotlib.pyplot as plt
from qiskit.circuit.quantumregister import AncillaRegister
from qiskit.quantum_info.states import statevector
import utility
import numpy as np
from qiskit.quantum_info import Statevector

class QuantumChannel:
    def __init__(self, error_rate=0.0) -> None:
        self._error_rate = error_rate
        self._state_vector = None

    def send_plain(self, state_vector):
        self._state_vector = state_vector

    def receive_plain(self):
        return self._state_vector

    def send(self, state_vector):
        qr_a = QuantumRegister(3, 'a')
        qr_b = QuantumRegister(3, 'b')
        qr_c = QuantumRegister(3, 'c')
        circ = QuantumCircuit(qr_a, qr_b, qr_c)
        circ.initialize(state_vector, [0])
        # outter layer protect phase flip
        circ.cnot(0, 3)
        circ.cnot(0, 6)
        circ.h(0)
        circ.h(3)
        circ.h(6)

        # innner layer protect bit flip
        circ += utility.bit_flip_encode_circ('a')
        circ += utility.bit_flip_encode_circ('b')
        circ += utility.bit_flip_encode_circ('c')

        # randomly apply X, Y, or Z gate to simulate transmission error
        if utility.quantum_bernoulli_bit(self._error_rate):
            dist = np.array([1.]*9)
            dist /= float(len(dist))
            error_bit = utility.quantum_random_choice(range(9), dist)
            dist = np.array([1.]*3)
            dist /= float(len(dist))
            error = utility.quantum_random_choice(['x', 'y', 'z'], dist)
            if error == 'x':
                # print('apply x to:', error_bit)
                circ.x(error_bit)
            elif error == 'y':
                # print('apply y to:', error_bit)
                circ.y(error_bit)
            elif error == 'z':
                # print('apply z to:', error_bit)
                circ.z(error_bit)
            else:
                raise NotImplementedError()
            
        # circ.draw('mpl') # debug
        self._state_vector = Statevector(circ)

    def receive(self):
        qr_a = QuantumRegister(3, 'a')
        qr_b = QuantumRegister(3, 'b')
        qr_c = QuantumRegister(3, 'c')
        ar = AncillaRegister(2, 'error_code_ar')
        cr = ClassicalRegister(2, 'error_code_cr')
        circ = QuantumCircuit(qr_a, qr_b, qr_c, ar, cr)
        circ.initialize(self._state_vector, range(9))

        circ += utility.bit_flip_recovery_circ('a')
        circ += utility.bit_flip_recovery_circ('b')
        circ += utility.bit_flip_recovery_circ('c')

        circ.h(0)
        circ.h(3)
        circ.h(6)
        circ.cnot(0, 9)
        circ.cnot(3, 9)
        circ.cnot(3, 10)
        circ.cnot(6, 10)
        circ.cnot(9, 0)
        circ.cnot(10, 6)
        circ.ccx(9, 10, 6)
        circ.ccx(9, 10, 3)
        circ.ccx(9, 10, 0)

        # unentangle the code qubits
        circ.cnot(0, 6)
        circ.cnot(0, 3)
        circ.measure([9, 10], [0, 1])

        # circ.draw('mpl') # debug
        backend = Aer.get_backend('statevector_simulator')
        job = backend.run(circ, shots=1)
        result = job.result()
        full_statevector = result.get_statevector()
        c_error_code, b_error_code, a_error_code, phase_error_code = [int(code, base=2) for code in (list(result.get_counts())[0]).split()]
        error_code = (phase_error_code << 9) + (a_error_code << 11) + (b_error_code << 13) + (c_error_code << 15)
        indices = [error_code, error_code + 1] # error codes in ancillas changes the index of states

        return full_statevector[indices]

    def send_3_code(self, state_vector):
        # entangle with 2 other qubits to correct X error
        circ = QuantumCircuit(3)
        circ.initialize(state_vector, [0])
        circ.cnot(0, 1)
        circ.cnot(0, 2)

        # randomly apply X gate to simulate transmission error
        if utility.quantum_bernoulli_bit(self._error_rate):
            dist = np.array([1., 1., 1.])
            dist /= float(len(dist))
            choice = utility.quantum_random_choice([0, 1, 2], dist)
            # print('apply x to:', choice) # debug
            circ.x(choice)
            # circ.draw('mpl') # debug

        self._state_vector = Statevector(circ)

    def receive_3_code(self):
        circ = QuantumCircuit(5, 2)
        circ.initialize(self._state_vector, [0, 1, 2])
        self._state_vector = None # non-clone theorem
        circ.cnot(0, 3)
        circ.cnot(1, 3)
        circ.cnot(1, 4)
        circ.cnot(2, 4)
        circ.cnot(3, 0)
        circ.cnot(4, 2)
        circ.ccx(3, 4, 0)
        circ.ccx(3, 4, 1)
        circ.ccx(3, 4, 2)

        # unentangle the code qubits
        circ.cnot(0, 2)
        circ.cnot(0, 1)
        circ.measure([3, 4], [0, 1])
        # circ.draw('mpl') # debug

        backend = Aer.get_backend('statevector_simulator')
        job = backend.run(circ, shots=1)
        result = job.result()
        full_statevector = result.get_statevector()
        error_code = int(list(result.get_counts())[0], base=2)
        indices = [error_code << 3, (error_code << 3) + 1]

        return full_statevector[indices]
