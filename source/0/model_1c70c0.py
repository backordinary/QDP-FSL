# https://github.com/abbysovich/QCB_Spring2021_project/blob/e71bdceb979be53d23aec70a2655cfc84cf4f1bb/model.py
#imports

import qiskit
import numpy as np
from qiskit.visualization import plot_histogram


# model

class Model:
    simulator = qiskit.Aer.get_backend('qasm_simulator')

    def draw_current_circuit(self):
        print(self.add_circuit.draw())

    # initialize 2 qubits
    def __init__(self):
        # TODO initialize with more friendly state vectors?
        self.circuit = qiskit.QuantumCircuit(2, 2)
        self.state1 = self.circuit.initialize(qiskit.quantum_info.random_statevector(2).data, 0)
        self.state2 = self.circuit.initialize(qiskit.quantum_info.random_statevector(2).data, 1)
        #self.state1, self.state2 = self.initialization([n for n in range(11)])
        self.add_circuit = qiskit.QuantumCircuit(2)

    def initialization(self, possible_states):
        indices = np.random.randint(0, len(possible_states)-1, 4)
        a1, a2 = 1/np.sqrt(indices[0]**2+indices[1]**2), 1/np.sqrt(indices[2]**2+indices[3]**2)
        vec1, vec2 = [indices[0], indices[1]], [indices[2], indices[3]]
        state1 = self.circuit.initialize([i*a1 for i in vec1], 0)
        state2 = self.circuit.initialize([i*a2 for i in vec2], 1)
        #state1 = self.circuit.initialize([1,0], 0)
        #state2 = self.circuit.initialize([1,0], 1)
        return state1, state2

    # Measure qubits and return state with max probability: ex. [0,1]
    def measureState(self):
        self.circuit.measure([0, 1], [0, 1])
        job = qiskit.execute(self.circuit, self.simulator, shots=1)  # 1 shot to keep it luck dependent?
        result = job.result()
        count = result.get_counts()
        # max_value = max(result.values())
        # return [k for k,v in count.items() if v==1][0]
        count = list(sorted(count.items(), key=lambda item: item[1], reverse=True))
        high = count[0][0]
        index = len(high)-1
        result = [int(high[index]), int(high[index-1])]
        #print(result)
        return result

    # Return a probability coefficient of specific state
    # state is an array of size 2 which contains 0 or 1
    # such as [0,1], [0,0], [1,0], [1,1]
    def getProbabilityOf(self, state):
        # to get state vector of qubit
        backend = qiskit.Aer.get_backend('statevector_simulator')
        result = qiskit.execute(self.circuit, backend).result()
        out_state = result.get_statevector()
        if state == [0, 0]:
            return out_state[0]
        elif state == [0, 1]:
            return out_state[1]
        elif state == [1, 0]:
            return out_state[2]
        elif state == [1, 1]:
            return out_state[3]
        else:
            assert False, 'bug!!'

    # Add a gate to the end of the circuit (at specified qubit)
    def add_unitary(self, name, qubit_no):
        if name == "H" or name == "h":
            self.circuit.h(qubit_no)
            self.add_circuit.h(qubit_no)
        elif name == "X" or name == "x":
            self.circuit.x(qubit_no)
            self.add_circuit.x(qubit_no)
        elif name == "Y" or name == "y":
            self.circuit.y(qubit_no)
            self.add_circuit.y(qubit_no)
        elif name == "Z" or name == "z":
            self.circuit.z(qubit_no)
            self.add_circuit.z(qubit_no)
        elif name == "I" or name == "i":
            self.circuit.id(qubit_no)
            self.add_circuit.id(qubit_no)

        #self.circuit += self.add_circuit

    def add_r_gate(self, parameter, qubit_no):
        self.circuit.rz(parameter, qubit_no)
        self.add_circuit.rz(parameter, qubit_no)

        #self.circuit += self.add_circuit

    def add_cnot(self, control_qubit_no, target_qubit_no):
        self.circuit.cx(control_qubit_no, target_qubit_no)
        self.add_circuit.cx(control_qubit_no, target_qubit_no)

        #self.circuit += self.add_circuit

    def num_qubits(self):
        return self.circuit.num_qubits

    def add_ancilla(self, num_ancilla):
        new_qc = qiskit.QuantumCircuit(self.num_qubits() + num_ancilla, 2)
        new_qc_two = qiskit.QuantumCircuit(self.num_qubits() + num_ancilla)
        self.circuit = new_qc.compose(self.circuit)
        self.add_circuit = new_qc_two.compose(self.add_circuit)