# https://github.com/phlie/someQs_code/blob/794d4eaa49ffd592c09a87a856f177aa09aa9554/research/and-of-two-arrays.py
# Creates two random Quantum arrays and then ANDS the two arrays

from qiskit import QuantumProgram, QISKitError, RegisterSizeError
import math
# Number of qubits and classical registers
num_qubits = 2                   # Number of qubits and classical registers
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
        qc.h(q_r[1])
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
        # print(result)
        # Get the two arrays and then do a bitwise AND comparison
        array_row_0 = result['01']
        array_row_1 = result['10']
        print(bin(array_row_0))
        print(bin(array_row_1))
        print(bin(array_row_0 & array_row_1))
# For errors in the circuit
except QISKitError as ex:
    print('There was an error in the circuit!. Error = {}'.format(ex))
except RegisterSizeError as ex:
    print("Error in number of registers!. Error = {}".format(ex))

