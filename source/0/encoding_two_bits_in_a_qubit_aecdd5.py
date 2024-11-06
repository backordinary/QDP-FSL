# https://github.com/phlie/someQs_code/blob/794d4eaa49ffd592c09a87a856f177aa09aa9554/research/encoding_two_bits_in_a_qubit.py
# A storage mechanism for storing two traditional bits in a Q

from qiskit import QuantumProgram, QISKitError, RegisterSizeError
import math
# Number of qubits and classical registers
num_qubits = 1                   # Number of qubits and classical registers
shots = 1000                    # Number of times the program should run
backend = 'local_qasm_simulator'  # Whether to use the simulator or the real thing
circuit_name = 'circuit'          # What you wish to call the circuit
loops = 1                         # The amount of times it loops over the whole program

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

# The lookup table is used to rotate around the 0~ axis in order to encode the two bits
rotation_lookup = {"00":0.0, "01":0.333, "10":0.666, "11":1.0}
bits = "00"
print("Input Bits: ", bits)
try:
    for i in range(loops):
        # Initializes the Program
        qp = QuantumProgram(specs=Q_SPECS)
        qc = qp.get_circuit(circuit_name)

        # Get both registers
        q_r = qp.get_quantum_register('qr')
        c_r = qp.get_classical_register('cr')

        # Circuit Design Goes here
        # Rotate the u3 gate by the amount of rotation setforth in the lookup table
        qc.u3(rotation_lookup[bits]*math.pi, 0, 0,q_r[0])

        # DECODER
        # Measure all the available qubits
        for qubit in range(num_qubits):
            qc.measure(q_r[qubit], c_r[qubit])

        # Compiles and executes the code
        out = qp.execute(circuit_name, backend=backend, shots=shots)

        # Get the results of the circuit
        result = out.get_counts(circuit_name)

        # The result array, loop through the following conditions and see what bits were in the message
        if '1' in result:
            if '0' in result:
                if result['0'] > result['1']:
                    print("Output Bits: ", '01')
                else:
                    print("Output Bits: ", '10')
            else:
                print("Output Bits: ", '11')
        else:
            print("Output Bits: ", '00')

        # The results section where you print out the information of the experiment
        print(result)

# For errors in the circuit
except QISKitError as ex:
    print('There was an error in the circuit!. Error = {}'.format(ex))
except RegisterSizeError as ex:
    print("Error in number of registers!. Error = {}".format(ex))

