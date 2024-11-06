# https://github.com/phlie/someQs_code/blob/794d4eaa49ffd592c09a87a856f177aa09aa9554/research/measuring_u3.py
# This program measures the probability of 1's over 0's if the u3 value is changed

from qiskit import QuantumProgram, QISKitError
import math
# Number of qubits and classical registers
num_qubits = 1
shots = 100000                    # Number of times the program should run
backend = 'local_qasm_simulator'  # Whether to use the simulator or the real thing
circuit_name = 'circuit'          # What you wish to call the circuit

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

print("|Frac | mul | result")
print("====================")
try:
    frac = 0.1 * float(1.0 + 0)
    frac = round(frac, 1)
    # Initializes the Program
    qp = QuantumProgram(specs=Q_SPECS)
    qc = qp.get_circuit(circuit_name)

    # Get both registers
    q_r = qp.get_quantum_register('qr')
    c_r = qp.get_classical_register('cr')

    # Circuit Design Goes here
    qc.u3(frac * math.pi, 0.0, 0.0, q_r[0])

    # Measure all the available qubits
    for qubit in range(num_qubits):
        qc.measure(q_r[qubit], c_r[qubit])
    
    # Compiles and executes the code
    out = qp.execute(circuit_name, backend=backend, shots=shots)

    # Get the results of the circuit
    result = out.get_counts(circuit_name)

    # The results section where you print out the information of the experiment
    print("|", frac, "| *pi |", result['1'] / result['0'])

        # For general errors, research later
except QISKitError as ex:
    print('There was an error in the circuit!. Error = {}'.format(ex))
try:
    frac = 0.1 * float(1.0 + 1)
    frac = round(frac, 1)
    # Initializes the Program
    qp = QuantumProgram(specs=Q_SPECS)
    qc = qp.get_circuit(circuit_name)

    # Get both registers
    q_r = qp.get_quantum_register('qr')
    c_r = qp.get_classical_register('cr')

    # Circuit Design Goes here
    qc.u3(frac * math.pi, 0.0, 0.0, q_r[0])

    # Measure all the available qubits
    for qubit in range(num_qubits):
        qc.measure(q_r[qubit], c_r[qubit])
    
    # Compiles and executes the code
    out = qp.execute(circuit_name, backend=backend, shots=shots)

    # Get the results of the circuit
    result = out.get_counts(circuit_name)

    # The results section where you print out the information of the experiment
    print("|", frac, "| *pi |", result['1'] / result['0'])

        # For general errors, research later
except QISKitError as ex:
    print('There was an error in the circuit!. Error = {}'.format(ex))
try:
    frac = 0.1 * float(1.0 + 2)
    frac = round(frac, 1)
    # Initializes the Program
    qp = QuantumProgram(specs=Q_SPECS)
    qc = qp.get_circuit(circuit_name)

    # Get both registers
    q_r = qp.get_quantum_register('qr')
    c_r = qp.get_classical_register('cr')

    # Circuit Design Goes here
    qc.u3(frac * math.pi, 0.0, 0.0, q_r[0])

    # Measure all the available qubits
    for qubit in range(num_qubits):
        qc.measure(q_r[qubit], c_r[qubit])
    
    # Compiles and executes the code
    out = qp.execute(circuit_name, backend=backend, shots=shots)

    # Get the results of the circuit
    result = out.get_counts(circuit_name)

    # The results section where you print out the information of the experiment
    print("|", frac, "| *pi |", result['1'] / result['0'])

        # For general errors, research later
except QISKitError as ex:
    print('There was an error in the circuit!. Error = {}'.format(ex))
try:
    frac = 0.1 * float(1.0 + 3)
    frac = round(frac, 1)
    # Initializes the Program
    qp = QuantumProgram(specs=Q_SPECS)
    qc = qp.get_circuit(circuit_name)

    # Get both registers
    q_r = qp.get_quantum_register('qr')
    c_r = qp.get_classical_register('cr')

    # Circuit Design Goes here
    qc.u3(frac * math.pi, 0.0, 0.0, q_r[0])

    # Measure all the available qubits
    for qubit in range(num_qubits):
        qc.measure(q_r[qubit], c_r[qubit])
    
    # Compiles and executes the code
    out = qp.execute(circuit_name, backend=backend, shots=shots)

    # Get the results of the circuit
    result = out.get_counts(circuit_name)

    # The results section where you print out the information of the experiment
    print("|", frac, "| *pi |", result['1'] / result['0'])

        # For general errors, research later
except QISKitError as ex:
    print('There was an error in the circuit!. Error = {}'.format(ex))
try:
    frac = 0.1 * float(1.0 + 4)
    frac = round(frac, 1)
    # Initializes the Program
    qp = QuantumProgram(specs=Q_SPECS)
    qc = qp.get_circuit(circuit_name)

    # Get both registers
    q_r = qp.get_quantum_register('qr')
    c_r = qp.get_classical_register('cr')

    # Circuit Design Goes here
    qc.u3(frac * math.pi, 0.0, 0.0, q_r[0])

    # Measure all the available qubits
    for qubit in range(num_qubits):
        qc.measure(q_r[qubit], c_r[qubit])
    
    # Compiles and executes the code
    out = qp.execute(circuit_name, backend=backend, shots=shots)

    # Get the results of the circuit
    result = out.get_counts(circuit_name)

    # The results section where you print out the information of the experiment
    print("|", frac, "| *pi |", result['1'] / result['0'])

        # For general errors, research later
except QISKitError as ex:
    print('There was an error in the circuit!. Error = {}'.format(ex))
try:
    frac = 0.1 * float(1.0 + 5)
    frac = round(frac, 1)
    # Initializes the Program
    qp = QuantumProgram(specs=Q_SPECS)
    qc = qp.get_circuit(circuit_name)

    # Get both registers
    q_r = qp.get_quantum_register('qr')
    c_r = qp.get_classical_register('cr')

    # Circuit Design Goes here
    qc.u3(frac * math.pi, 0.0, 0.0, q_r[0])

    # Measure all the available qubits
    for qubit in range(num_qubits):
        qc.measure(q_r[qubit], c_r[qubit])
    
    # Compiles and executes the code
    out = qp.execute(circuit_name, backend=backend, shots=shots)

    # Get the results of the circuit
    result = out.get_counts(circuit_name)

    # The results section where you print out the information of the experiment
    print("|", frac, "| *pi |", result['1'] / result['0'])

        # For general errors, research later
except QISKitError as ex:
    print('There was an error in the circuit!. Error = {}'.format(ex))
try:
    frac = 0.1 * float(1.0 + 6)
    frac = round(frac, 1)
    # Initializes the Program
    qp = QuantumProgram(specs=Q_SPECS)
    qc = qp.get_circuit(circuit_name)

    # Get both registers
    q_r = qp.get_quantum_register('qr')
    c_r = qp.get_classical_register('cr')

    # Circuit Design Goes here
    qc.u3(frac * math.pi, 0.0, 0.0, q_r[0])

    # Measure all the available qubits
    for qubit in range(num_qubits):
        qc.measure(q_r[qubit], c_r[qubit])
    
    # Compiles and executes the code
    out = qp.execute(circuit_name, backend=backend, shots=shots)

    # Get the results of the circuit
    result = out.get_counts(circuit_name)

    # The results section where you print out the information of the experiment
    print("|", frac, "| *pi |", result['1'] / result['0'])

        # For general errors, research later
except QISKitError as ex:
    print('There was an error in the circuit!. Error = {}'.format(ex))
try:
    frac = 0.1 * float(1.0 + 7)
    frac = round(frac, 1)
    # Initializes the Program
    qp = QuantumProgram(specs=Q_SPECS)
    qc = qp.get_circuit(circuit_name)

    # Get both registers
    q_r = qp.get_quantum_register('qr')
    c_r = qp.get_classical_register('cr')

    # Circuit Design Goes here
    qc.u3(frac * math.pi, 0.0, 0.0, q_r[0])

    # Measure all the available qubits
    for qubit in range(num_qubits):
        qc.measure(q_r[qubit], c_r[qubit])
    
    # Compiles and executes the code
    out = qp.execute(circuit_name, backend=backend, shots=shots)

    # Get the results of the circuit
    result = out.get_counts(circuit_name)

    # The results section where you print out the information of the experiment
    print("|", frac, "| *pi |", result['1'] / result['0'])

        # For general errors, research later
except QISKitError as ex:
    print('There was an error in the circuit!. Error = {}'.format(ex))
try:
    frac = 0.1 * float(1.0 + 8)
    frac = round(frac, 1)
    # Initializes the Program
    qp = QuantumProgram(specs=Q_SPECS)
    qc = qp.get_circuit(circuit_name)

    # Get both registers
    q_r = qp.get_quantum_register('qr')
    c_r = qp.get_classical_register('cr')

    # Circuit Design Goes here
    qc.u3(frac * math.pi, 0.0, 0.0, q_r[0])

    # Measure all the available qubits
    for qubit in range(num_qubits):
        qc.measure(q_r[qubit], c_r[qubit])
    
    # Compiles and executes the code
    out = qp.execute(circuit_name, backend=backend, shots=shots)

    # Get the results of the circuit
    result = out.get_counts(circuit_name)

    # The results section where you print out the information of the experiment
    print("|", frac, "| *pi |", result['1'] / result['0'])

        # For general errors, research later
except QISKitError as ex:
    print('There was an error in the circuit!. Error = {}'.format(ex))
