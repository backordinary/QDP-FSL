# https://github.com/phlie/someQs_code/blob/794d4eaa49ffd592c09a87a856f177aa09aa9554/research/random-number-generator.py
# This program generates a random number

from qiskit import QuantumProgram, QISKitError
import math, random
# Number of qubits and classical registers
num_qubits = 1
shots = 10000                  # Number of times the program should run
backend = 'local_qasm_simulator'  # Whether to use the simulator or the real thing
circuit_name = 'circuit'          # What you wish to call the circuit


num_rand_digits = 2
num_runs = 2000
digit_array = [0]*num_rand_digits

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

print("Shots:", shots, " Runs: ", num_runs)

for run in range(num_runs):
    try:
        # frac = 0.6 + float(random.randint(0, 9))/100.0 + float(random.randint(0, 9))/1000.0
        frac = 0.6
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

        random_number = (result['1']/result['0']) * (10.0 ** 18)
        random_number = round(random_number, 0)
        string_random_number = str(format(random_number, 'f'))

        for i in range(4,16):
            rand_sum = int(string_random_number[i])
        if len(string_random_number) <= 16:
            number = random.randint(0, num_rand_digits)
            print("Weird number, RandInt: ", number, " Frac: ", frac, "RN: ", string_random_number)
        else:
            number = rand_sum % num_rand_digits


        # The results section where you print out the information of the experimentX1
        # print("RandInt: ", number, " Frac: ", frac, "RN: ", string_random_number)
        digit_array[int(number)] += 1

        if run % 10 == 0:
            print("Run: ", run)
            print(digit_array)

        # For general errors, research later
    except QISKitError as ex:
        print('There was an error in the circuit!. Error = {}'.format(ex))

print("======================================")
print(digit_array)
print("======================================")
