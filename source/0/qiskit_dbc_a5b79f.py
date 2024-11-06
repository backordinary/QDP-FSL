# https://github.com/iamsmurph/es170_ml_project/blob/f78502cac4852b8ac94bd7486e0b52148e8e69cf/ml/qiskit_dbc.py
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, BasicAer
import numpy as np
'''
This Module is exactly like the pyquil dbc_module but with the original qiskit,
taken from the original repository - changes made to the classify and angles function
others are largely the same.
'''
class QK_DistanceBasedClassifier:

    def initialize_registers(self, num_registers):
        self.q = QuantumRegister(4)
        self.c = ClassicalRegister(4)
        # name the individual qubits for more clarity
        self.ancilla_qubit = self.q[0]
        self.index_qubit = self.q[1]
        self.data_qubit = self.q[2]
        self.class_qubit = self.q[3]

    def create_circuit(self, angles):


        qc = QuantumCircuit(self.q, self.c)
        qc.h(self.ancilla_qubit)
        qc.h(self.index_qubit)
        qc.cx(self.ancilla_qubit, self.data_qubit)
        qc.u3(-angles[0], 0, 0, self.data_qubit)
        qc.cx(self.ancilla_qubit, self.data_qubit)
        qc.u3(angles[0], 0, 0, self.data_qubit)
        qc.barrier()
        qc.x(self.ancilla_qubit)
        qc.barrier()
        qc.ccx(self.ancilla_qubit, self.index_qubit, self.data_qubit)
        qc.barrier()
        qc.x(self.index_qubit)
        qc.barrier()
        qc.ccx(self.ancilla_qubit, self.index_qubit, self.data_qubit)
        qc.cx(self.index_qubit, self.data_qubit)
        qc.u3(angles[1], 0, 0, self.data_qubit)
        qc.cx(self.index_qubit, self.data_qubit)
        qc.u3(-angles[1], 0, 0, self.data_qubit)
        qc.ccx(self.ancilla_qubit, self.index_qubit, self.data_qubit)
        qc.cx(self.index_qubit, self.data_qubit)
        qc.u3(-angles[1], 0, 0, self.data_qubit)
        qc.cx(self.index_qubit, self.data_qubit)
        qc.u3(angles[1], 0, 0, self.data_qubit)
        qc.barrier()
        qc.cx(self.index_qubit, self.class_qubit)
        qc.barrier()
        qc.h(self.ancilla_qubit)
        qc.barrier()
        qc.measure(self.q, self.c)

        return qc

    def simulate(self, quantum_circuit):

        # noisy simulation
        backend_sim = BasicAer.get_backend('qasm_simulator')
        job_sim = execute(quantum_circuit, backend_sim)

        # retrieve the results from the simulation
        return job_sim.result()

    def get_angle_for_vec(self, vec):
        return float(np.arccos(vec[0]))*2.0

    def interpret_results(self, result_counts):
        total_samples = sum(result_counts.values())

        # define lambda function that retrieves only results where the ancilla is in the |0> state
        # here it checks the last element, if 0 it stays, otherwise it goes.
        post_select = lambda counts: [(state, occurences) for state, occurences in counts.items() if state[-1] == '0']

        # perform the postselection
        postselection = dict(post_select(result_counts))
        postselected_samples = sum(postselection.values())

        retrieve_class = lambda binary_class: [occurences for state, occurences in postselection.items() if state[0] == str(binary_class)]
        prob_class0 = sum(retrieve_class(0))/postselected_samples
        prob_class1 = sum(retrieve_class(1))/postselected_samples


        return prob_class0, prob_class1

    def classify(self, test_vector, training_set):

        training_vectors = [tuple_[0] for tuple_ in training_set]
        angles = []
        angles.append(self.get_angle_for_vec(test_vector)/2)
        angles.append(self.get_angle_for_vec(training_vectors[1])/4)

        # initialize the Q and C registers
        self.initialize_registers(num_registers=4)
        # create the quantum circuit
        qc = self.create_circuit(angles=angles)

        # simulate and get the results
        result = self.simulate(qc)

        prob_class0, prob_class1 = self.interpret_results(result.get_counts(qc))

        if prob_class0 > prob_class1:
            return 0
        elif prob_class0 < prob_class1:
            return 1
        else:
            return 2
