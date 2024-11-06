# https://github.com/Wirkungsfunktional/QuantumInformationProcessing/blob/b3929f9a8e838e3dbda9feee5cf753d1204e75a1/QuantumGates/QuantumAlgorithm.py
try:
    import qiskit
    from qiskit import QuantumProgram
except ImportError:
    print("Module qiskit does not exists")
    exit()
if (qiskit.__version__ != "0.4.8"):
    print("Module version of qiskit is not tested. There could occure Problems.")

import numpy as np



class QuantumAlgorithm:
    def __init__(self, name, N_qm, N_cl):
        self.name = name
        self.N_qm = N_qm
        self.N_cl = N_cl
        self.number_of_gates = 0
        self.input_state = [0]*self.N_qm
        self.quantum_program = QuantumProgram()
        self.quantum_register = self.quantum_program.create_quantum_register(
                    'quantum_register', self.N_qm)
        self.classical_register = self.quantum_program.create_classical_register(
                    'classical_register', self.N_cl)
        self.quantum_circuit = self.quantum_program.create_circuit(
                                                    self.name,
                                                    [self.quantum_register],
                                                    [self.classical_register])
        self.result = None

    def invert_nth_qbit(self, n):
        """Invert the qbit at position n. Invert means 0 to 1 and 1 to 0. This
        is done by using the pauli_x matrix."""
        assert n >= 0 , "negativ index"
        assert n < self.N_qm, "index above size of register"
        self.quantum_circuit.x(self.quantum_register[n])
        self.input_state[n] = self.input_state[n] ^ 1 #Inversion by XOR

    def set_nth_qbit(self, n, theta):
        self.quantum_circuit.ry(theta, self.quantum_register[n])

    def get_input_state(self):
        return self.input_state


    def measure_nth_qbit(self, n):
        assert n >= 0 , "negativ index"
        assert n < self.N_qm, "index above size of quantum register"
        assert n < self.N_cl, "index above size of classical register"
        self.quantum_circuit.measure(   self.quantum_register[n],
                                        self.classical_register[n])

    def run(self):
        """Execute the predescribed quantum circuit."""
        self.result = self.quantum_program.execute(self.name)

    def get_result(self):
        return self.result.get_counts(self.name)

    def get_number_of_gates(self):
        return self.number_of_gates

    def add_gate(self, name, register_list, *arg):
        self.number_of_gates += 1
        if name == "pauli_x":
            self.quantum_circuit.x(self.quantum_register[register_list[0]])
        elif name == "pauli_y":
            self.quantum_circuit.y(self.quantum_register[register_list[0]])
        elif name == "pauli_z":
            self.quantum_circuit.z(self.quantum_register[register_list[0]])
        elif name == "hadamard":
            self.quantum_circuit.h( self.quantum_register[register_list[0]])
        elif name == "CNOT":
            self.quantum_circuit.cx(    self.quantum_register[register_list[0]],
                                        self.quantum_register[register_list[1]])
        elif name == "rotate_y":
            self.quantum_circuit.ry(    arg[0],
                                        self.quantum_register[register_list[0]])
