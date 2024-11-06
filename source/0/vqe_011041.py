# https://github.com/pchung39/quantum_variational_eigensolver/blob/339dcb928ed1f1624d56d91552031fcbe5a2e2d6/vqe.py
import numpy as np
from itertools import product
from operator import matmul
import functools
from qiskit import *
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.algorithms import NumPyEigensolver
from qiskit.providers.aer import noise
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
import json

class VQE(object): 

    def __init__(self, angle_range, shots=1000, noise=None):
        self.angle_range = angle_range
        self.angle = 0
        self.shots = shots
        self.noise = noise
        self.vqe_info = {
            "estimated_energy": {},
            "simulator_results": {},
            "decomposed_hamiltonian": {}
        }

    def return_vqe_info(self):

        return self.vqe_info


    def run_classical_eigensolver(self, a, b, c, d):
        """
        Creates Hamiltonian operator and runs a classical eigensolver. Providing an numeric value for a,b,c,d 
        this function will create a hamiltonian for the paul matrices [II, ZZ, XX, YY] and calculate the ground state 
        using a classical eigensolver.    
        """

        pauli_dict = {
            'paulis': [{"coeff": {"imag": 0.0, "real": a}, "label": "II"},
                    {"coeff": {"imag": 0.0, "real": b}, "label": "ZZ"},
                    {"coeff": {"imag": 0.0, "real": c}, "label": "XX"},
                    {"coeff": {"imag": 0.0, "real": d}, "label": "YY"}
                    ]
        }
        hamiltonian = WeightedPauliOperator.from_dict(pauli_dict)
        exact_result = NumPyEigensolver(hamiltonian).run()
        reference_energy = min(np.real(exact_result.eigenvalues))
        print('The exact ground state energy is: {}'.format(reference_energy))    
        self.vqe_info["classical_ground_state"] = exact_result


    def decompose_hamiltonian(self, hamiltonian):
        '''
        This method provides a utility to decompose any Hermitian matrix into constituent Pauli matrices.
        Helpful to know what coefficient values must be set for calculating the weights of each measurement axis. 
        '''

        pauli_dictionary = {
            "I": np.array([[1,  0], [0,  1]], dtype=complex),
            "X": np.array([[0,  1], [1,  0]], dtype=complex),
            "Y": np.array([[0, -1j], [1j,  0]], dtype=complex),
            "Z": np.array([[1,  0],[0, -1]], dtype=complex) 
        }

        pauli_coefficients = {}

        for matrix in ["I", "X", "Y", "Z"]:
            matrix_entry = pauli_dictionary[matrix]
            pauli_matrix_tensored = np.kron(matrix_entry, matrix_entry)
            pauli_coefficients["{}".format(matrix+matrix)] = str(0.25 * np.trace(np.matmul(pauli_matrix_tensored, hamiltonian)))

        self.vqe_info["decomposed_hamiltonian"] = pauli_coefficients
        
        return pauli_coefficients

    def generate_circuit(self, measurement_axis):
        '''
        This method is responsible for generating the quantum circuit that will execute the vqe algorithm. 

        '''
        if measurement_axis not in ["X", "Y", "Z"]:
            raise Exception("Unknown measurement axis was provided: {}".format(measurement_axis))

        q = QuantumRegister(2)
        c = ClassicalRegister(2)
        circuit = QuantumCircuit(q, c)  

        '''
        Ansatz
        '''

        circuit.h(0)
        circuit.cx(0,1)
        circuit.rx(self.angle, 0)
        circuit.barrier()

        '''
        Pre-measurement rotations (For calculating in X and Y axis)
        '''
        if measurement_axis == 'X':
            circuit.ry(-np.pi/2, q[0])
            circuit.ry(-np.pi/2, q[1])
        elif measurement_axis == 'Y':
            circuit.rx(np.pi/2, q[0])
            circuit.rx(np.pi/2, q[1])
        else:
            pass

        '''
        Measure and store state in classical bits
        '''
        circuit.barrier()
        circuit.measure(q[0], c[0])
        circuit.measure(q[1], c[1])

        return circuit  


    def execute_measurement(self, circuit):
        ''' 
        With the quantum circuit assembled, this method is responsible for executing
        the quantum simulator. Note that if self.noise is not None, it will run
        the simulator with noise.
        '''

        if self.noise:
            backend = Aer.get_backend('qasm_simulator')
            job = execute(circuit, backend, shots=self.shots, 
                          basis_gates=self.noise.basis_gates, noise_model=self.noise)
        else:
            backend = Aer.get_backend('qasm_simulator')
            job = execute(circuit, backend, shots=self.shots)
            
        result = job.result()
        return result


    def calculate_expectation_values(self, measurement_results, axis):
        '''
        Provided results from the quantum simulator, this method calculates the expectation value.
        '''

        counts = measurement_results.get_counts()
        self.vqe_info["simulator_results"][str(self.angle)] = {}
        self.vqe_info["simulator_results"][str(self.angle)][axis] = counts

        counts_00 = counts["00"] if "00" in counts else 0
        counts_01 = counts["01"] if "01" in counts else 0
        counts_10 = counts["10"] if "10" in counts else 0
        counts_11 = counts["11"] if "11" in counts else 0

        expected_value = (counts_00 - counts_01 - counts_10 + counts_11) / self.shots

        return expected_value

    def measure_hamiltonian(self, axis):

        '''
        This method is a wrapper around the functions that need to be executed in order to 
        produce an expectation value for the given pauli matrix
        '''

        circuit = self.generate_circuit(axis)
        measurement_results = self.execute_measurement(circuit)
        expectation_value = self.calculate_expectation_values(measurement_results, axis)

        return expectation_value

    def get_energy_state(self, angle):

        '''
        After all states have been measured, this method will sum each measurement and return
        the energy state
        '''

        self.angle = float(angle)
        x_measurement = -0.5 * (self.measure_hamiltonian("X"))
        y_measurement = -0.5 * (self.measure_hamiltonian("Y"))
        z_measurement = 0.5 * (self.measure_hamiltonian("Z"))
        i_measurement = 0.5 * 1

        energy_state = x_measurement + y_measurement + z_measurement + i_measurement
        
        return energy_state


    def find_ground_state(self):

        '''
        This is our main function to invoke which will execute all necessary methods to calculate
        the lowest eigenvalue (ground state) of the matrix.
        '''

        measurements = []

        for angle in self.angle_range: 
            measurement = self.get_energy_state(angle)
            measurements.append(measurement)
            self.vqe_info["estimated_energy"][str(angle)] = measurement

        ground_state_theta = min(self.vqe_info["estimated_energy"], key=self.vqe_info["estimated_energy"].get)
        ground_state = min(measurements)

        self.vqe_info["vqe_ground_state"] = ground_state
        self.vqe_info["ground_state_theta"] = ground_state_theta

        print("Ground state: {}".format(ground_state))
        print("Angle (theta) for ground state: ", ground_state_theta)



if __name__ == "__main__":
    vqe = VQE(angle_range=np.linspace(0, 2 * np.pi, 20))
    vqe.find_ground_state()

