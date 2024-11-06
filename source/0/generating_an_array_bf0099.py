# https://github.com/phlie/someQs_code/blob/794d4eaa49ffd592c09a87a856f177aa09aa9554/research/generating-an-array.py
# Creates a byte from the superposition of a Qubit

from qiskit import QuantumProgram, QISKitError, RegisterSizeError
import math
# Number of qubits and classical registers
num_qubits = 1                   # Number of qubits and classical registers
shots = 255                    # Number of times the program should run
backend = 'local_qasm_simulator'  # Whether to use the simulator or the real thing
circuit_name = 'circuit'          # What you wish to call the circuit
loops = 1

# This is where the quantum and classical registers are defined
Q_SPECS = {
    'circuits': [{
        'name': circuit_name,
        'quantum_registers': [{
            'name': 'qr',
            'size': num_qubits
        }],
        'classical_registers': [{
            'name': 'cr',
            'size': num_qubits
        }]}],
}

try:
    for i in range(loops):
        # Initializes the Program
        qp = QuantumProgram(specs=Q_SPECS)
        qc = qp.get_circuit(circuit_name)

        # Get both registers
        q_r = qp.get_quantum_register('qr')
        c_r = qp.get_classical_register('cr')

        # Circuit Design Goes here
        qc.h(q_r[0])

        # qc.cx(q_r[1], q_r[2])  # CNOT from 1 to 2
        # qc.x(q_r[1])

        # Measure all the available qubits
        for qubit in range(num_qubits):
            qc.measure(q_r[qubit], c_r[qubit])

            # Compiles and executes the code
            out = qp.execute(circuit_name, backend=backend, shots=shots)

            # Get the results of the circuit
            result = out.get_counts(circuit_name)
            
            # The results section where you print out the information of the experiment
            # Prints a randomly created byte array
            print(bin(result['1']))
# For errors in the circuit
except QISKitError as ex:
    print('There was an error in the circuit!. Error = {}'.format(ex))
except RegisterSizeError as ex:
    print("Error in number of registers!. Error = {}".format(ex))

