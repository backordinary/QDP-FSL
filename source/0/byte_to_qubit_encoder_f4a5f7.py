# https://github.com/phlie/someQs_code/blob/794d4eaa49ffd592c09a87a856f177aa09aa9554/research/byte_to_qubit_encoder.py
# This is a quantum program that encodes as many bits as it can on one Q into a byte
# It uses a helper function to generate all the nessasary lookup tables and the
# bit array with all possible combitations. It can be modified to store up to 8 bits
# on one Q, to modify the program just change the top values


from qiskit import QuantumProgram, QISKitError, RegisterSizeError
import math                     # Used for PI
from random import randint      # Imports the random integer function used to grab the bits
import helper.get_nth_qubit as gnq  # The helper function used in the Quantum Decoding
import helper.bit_decoder as bd

# Number of qubits and classical registers
num_qubits = 3                   # Number of qubits and classical registers
# The total amount of superpositions. Is possible to change but the lower the value, the less accuracy
shots = 10000                    # Number of times the program should run
circuit_name = 'circuit'          # What you wish to call the circuit
loops = 1                        # The amount of times it loops over the whole program
num_bits_in_byte = 4
backend = 'local_qasm_simulator'  # Whether to use the simulator or the real thingn

# For running on the actual quantum computer
# backend = 'ibmqx5'
# import helper.Qconfig as Qconfig

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
rotation_lookup = bd.generate_lookup(num_bits_in_byte)

# All of the possible bits that can be encoded and decoded
bits_to_test = bd.init_generate_bits(num_bits_in_byte)

# print(rotation_lookup)
# print(bits_to_test)

quantum_word = ''       # Used to combine the results
total_input = ''        # Used to gather all the inputs
try:
    for i in range(loops):
        bits = []
        # Initializes the Program
        # Get the bits to test for this iteration, 4 bits for each itteration
        for x in range(num_qubits):
            bits.append(bits_to_test[randint(0,2**num_bits_in_byte - 1)])  # Append the byte for the test to bits
        # print(len(bits_to_test))
        qp = QuantumProgram(specs=Q_SPECS)

        # For running the code on IBM's Q
        # qp.set_api(Qconfig.APItoken, Qconfig.config['url'])

        qc = qp.get_circuit(circuit_name)

        # Get both registers
        q_r = qp.get_quantum_register('qr')
        c_r = qp.get_classical_register('cr')

        # ENCODER
        # Rotate the u3 gate by the amount perscribed in the lookup table for each Q
        print("B: ", bits)
        for x in range(num_qubits):
            qc.u3(rotation_lookup[bits[x]], 0, 0,q_r[x])
        qc.barrier()            # Preserves the Quantum Superposition for longer (I believe)

        # DECODER
        # Measure all the available qubits
        for q in range(num_qubits):
            qc.measure(q_r[q], c_r[q])

        # Compiles and executes the code
        out = qp.execute(circuit_name, backend=backend, shots=shots)

        # Get the results of the circuit
        result = out.get_counts(circuit_name)

        output_bits = ''        # Saves the output
        print("Result: ", result)

        for current_qubit in range(num_qubits):  # For each Q in the circuit, loop through
            possible_results = []                # We want to initialize this to nil
            current_bit_result = 0;              # Each itteration start fresh
            # Use the helper function to get an array of possible arrangements of the classical registers
            possible_results = gnq.init_bit_find(num_qubits, current_qubit)
            # print("PR: ", possible_results)

            # For all the possible results, loop through and ...
            for r in range(len(possible_results)):
                if possible_results[r] in result:  # If the current possible result is in results
                    current_bit_result += result[possible_results[r]]  # Add to the total 1's for that Q
            # print("CBR: ", current_bit_result)
            # This pat gets the correct output bits without knowing the input bits by using the difference
            # between the amount of 1's and 0's to statistically say what the ouput bits are
            # If the total 1's is less than 20 with 10,000 shots, output '0000'
            output_bits = bd.decoder(current_bit_result, shots, num_bits_in_byte)

            print("Output Bits: ", output_bits)  # Prints the output bits for the current Q
            if bits[current_qubit] == output_bits:  # Checks to see which is true
                print("Case ", bits[current_qubit], "is TRUE")
            else:
                print("Case ", bits[current_qubit], "is FALSE")
                print("FALSE")
                print("FALSE")
                print("FALSE")

            print()
            quantum_word += output_bits + ", "  # Combine the results to get the quantum word


        for b in range(len(bits)):  # Loop through and make the input word
            total_input += bits[b] + ", "
    print("Input:  ", total_input)
    print("Output: ", quantum_word)

# For errors in the circuit
except QISKitError as ex:
    print('There was an error in the circuit!. Error = {}'.format(ex))
except RegisterSizeError as ex:
    print("Error in number of registers!. Error = {}".format(ex))
