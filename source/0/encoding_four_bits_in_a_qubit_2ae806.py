# https://github.com/phlie/someQs_code/blob/794d4eaa49ffd592c09a87a856f177aa09aa9554/research/encoding_four_bits_in_a_qubit.py
# This program encodes 4 bits and then decodes them based on the difference betweeen
# the amount of 1's and 0's, it holds 4 bits on a single Q, and then only uses a classical
# gate to decode them

from qiskit import QuantumProgram, QISKitError, RegisterSizeError
import math
# Number of qubits and classical registers
num_qubits = 1                   # Number of qubits and classical registers
# The total amount of superpositions. Is possible to change but the lower the value, the less accuracy
shots = 10000                    # Number of times the program should run
backend = 'local_qasm_simulator'  # Whether to use the simulator or the real thing
circuit_name = 'circuit'          # What you wish to call the circuit
loops = 16                         # The amount of times it loops over the whole program

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
# The rotation lookup table gives the percents to rotate the u3 gate by which are just divisions of 16
rotation_lookup = {"0000":0.0, "0001":0.0625, "0010":0.125, "0011":0.1875, "0100":0.25, "0101":0.3125, "0110":0.375, "0111":0.4375, "1000":0.5, "1001":0.5625, "1010":0.625, "1011":0.6875, "1100":0.75, "1101":0.8125, "1110":0.875, "1111":0.9375, 'X': 1}

# All of the possible bits that can be encoded and decoded
bits_to_test = ["0000", "0001", "0010", "0011", "0100", "0101", "0110", "0111", "1000", "1001", "1010", "1011", "1100", "1101", "1110", "1111"]
try:
    for i in range(loops):
        # Initializes the Program
        # Get the bits to test for this iteration, 4 bits for each itteration
        bits = bits_to_test[i]
        print("Input Bits:  ", bits)

        qp = QuantumProgram(specs=Q_SPECS)
        qc = qp.get_circuit(circuit_name)

        # Get both registers
        q_r = qp.get_quantum_register('qr')
        c_r = qp.get_classical_register('cr')

        # ENCODER
        # Rotate the u3 gate by the amount perscribed in the lookup table
        qc.u3(rotation_lookup[bits]*math.pi, 0, 0,q_r[0])

        # DECODER
        # Measure all the available qubits
        for qubit in range(num_qubits):
            qc.measure(q_r[qubit], c_r[qubit])

        # Compiles and executes the code
        out = qp.execute(circuit_name, backend=backend, shots=shots)

        # Get the results of the circuit
        result = out.get_counts(circuit_name)

        # This part gets the correct output bits without knowing the input bits by using the difference
        # between the amount of 1's and 0's to statistically say what the ouput bits are
        if '1' in result:
            # If the total 1's is less than 20 with 10,000 shots, output '0000'
            if result['1'] < 20 * shots / 10000 :
                output_bits = "0000"  # Save the output bits for later
                print("Output Bits: ", "0000")
            elif result['1'] < 225 * shots / 10000:
                output_bits = "0001"
                print("Output Bits: ", "0001")
            elif result['1'] < 600 * shots / 10000:
                output_bits = "0010"
                print("Output Bits: ", "0010")
            elif result['1'] < 1250 * shots / 10000:
                output_bits = "0011"
                print("Output Bits: ", "0011")
            elif result['1'] < 1875 * shots / 10000:
                output_bits = "0100"
                print("Output Bits: ", "0100")
            elif result['1'] < 2725 * shots / 10000:
                output_bits = "0101"
                print("Output Bits: ", "0101")
            elif result['1'] < 3600 * shots / 10000:
                output_bits = "0110"
                print("Output Bits: ", "0110")
            elif result['1'] < 4500 * shots / 10000:
                output_bits = "0111"
                print("Output Bits: ", "0111")
            elif result['1'] < 5500 * shots / 10000:
                output_bits = "1000"
                print("Output Bits: ", "1000")
            elif result['1'] < 6400 * shots / 10000:
                output_bits = "1001"
                print("Output Bits: ", "1001")
            elif result['1'] < 7275 * shots / 10000:
                output_bits = "1010"
                print("Output Bits: ", "1010")
            elif result['1'] < 8125 * shots / 10000:
                output_bits = "1011"
                print("Output Bits: ", "1011")
            elif result['1'] < 8750 * shots / 10000:
                output_bits = "1100"
                print("Output Bits: ", "1100")
            elif result['1'] < 9400 * shots / 10000:
                output_bits = "1101"
                print("Output Bits: ", "1101")
            elif result['1'] < 9775 * shots / 10000:
                output_bits = "1110"
                print("Output Bits: ", "1110")
            else:
                output_bits = "1111"
                print("Output Bits: ", "1111")
        else:
            output_bits = "0000"
            print("Output Bits: ", "0000")  # In the 0 case

        if bits == output_bits:
            print("Case ", bits, "is TRUE")
        else:
            print("Case ", bits, "is FALSE")

        print()

# For errors in the circuit
except QISKitError as ex:
    print('There was an error in the circuit!. Error = {}'.format(ex))
except RegisterSizeError as ex:
    print("Error in number of registers!. Error = {}".format(ex))
