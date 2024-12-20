# https://github.com/joeyp722/Enigma/blob/d586f6c0411bc2b2b0d2a5c8d525fb3823d837ed/enigma/quantum_algorithms.py
# Defining several quantum algorithms
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, assemble, IBMQ, Aer, BasicAer, execute
from qiskit.circuit.library.standard_gates import HGate, XGate, ZGate
from qiskit.visualization import plot_histogram

from math import sqrt, pi, log, ceil
import random as rd


import enigma.quantum_gates as qg
import enigma.sat as sat

# Grover sat solver function.
def grover_sat_solver(cnf, lamb, shots):

    # Set iterations upper bound.
    m = 1

    # Determine literals.
    literals = []
    for j in range(len(cnf)):
        literals = list(set(literals) | set([abs(i) for i in cnf[j]]))

    # Repeating the cnf grover solver for different number of iterations.
    while True:

        # Get required qubits.
        qubits = list(range(0, qg.req_qubits_oracle(cnf, False, len(cnf)*[1])))

        # Get the number of iterations for this repetition.
        iterations = rd.randint(1, round(m))

        # Define registers.
        qreg = QuantumRegister(len(qubits))
        qc = QuantumCircuit(qreg)

        # Determine literal qubits.
        literal_qubits = []
        for j in range(len(literals)):
            literal_qubits.append(j)

        # Hadamards for literal qubits.
        for i in range(len(literal_qubits)):
            qc.h(i)

        # Bit flip for phase kickback qubit.
        qc.x(qubits[-1])

        # Grover gate for cnf.
        cnf_grover = qg.cnf_grover(cnf, iterations)
        qc.append(cnf_grover, qubits)

        # Reset bit flip for phase kickback qubit for ease of use of statevector
        qc.x(qubits[-1])

        # # Measure all the qubits.
        qc.measure_all()

        # Get backend.
        backend = BasicAer.get_backend('qasm_simulator')

        # Execute job.
        job = execute(qc, backend, shots = shots)

        # Get counts form job.
        counts = job.result().get_counts()

        # Creating iteratable object.
        count_iterator = iter(counts)

        # Iterate thru the counts.
        for j in range(len(counts)):

            # Going to the next measurent.
            measurement = next(count_iterator)

            # Constructing propositional solution.
            solution = []
            for i in range(1,len(literals)+1):

                # Converting the measurent string to solution array.
                solution.append(literals[i-1]) if bool(int(measurement[-i])) else solution.append(-literals[i-1])

            # Return the proposed solution if correct.
            if sat.verify(cnf, solution): return solution

            # Return None if no solution was found.
            if m >= sqrt(2^len(literal_qubits)): return None

            # Increment iterations upper bound.
            m = min(lamb*m , sqrt(2^len(literal_qubits)))

# Grover sat solver function with adder oracle.
def grover_sat_solver_adder(cnf, number, compare_string, lamb, shots):

    # Set iterations upper bound.
    m = 1

    # Determine literals.
    literals = []
    for j in range(len(cnf)):
        literals = list(set(literals) | set([abs(i) for i in cnf[j]]))

    # Repeating the cnf grover solver for different number of iterations.
    while True:

        # Get required qubits.
        qubits = list(range(0, qg.req_qubits_oracle(cnf, True, number)))

        # Get the number of iterations for this repetition.
        iterations = rd.randint(1, round(m))

        # Define registers.
        qreg = QuantumRegister(len(qubits))
        qc = QuantumCircuit(qreg)

        # Determine literal qubits.
        literal_qubits = []
        for j in range(len(literals)):
            literal_qubits.append(j)

        # Hadamards for literal qubits.
        for i in range(len(literal_qubits)):
            qc.h(i)

        # Bit flip for phase kickback qubit.
        qc.x(qubits[-1])

        # Grover gate for cnf.
        cnf_grover = qg.cnf_grover_adder(cnf, number, compare_string, iterations)
        qc.append(cnf_grover, qubits)

        # Reset bit flip for phase kickback qubit for ease of use of statevector
        qc.x(qubits[-1])

        # # Measure all the qubits.
        qc.measure_all()

        # Get backend.
        backend = BasicAer.get_backend('qasm_simulator')

        # Execute job.
        job = execute(qc, backend, shots = shots)

        # Get counts form job.
        counts = job.result().get_counts()

        # Creating iteratable object.
        count_iterator = iter(counts)

        # Iterate thru the counts.
        for j in range(len(counts)):

            # Going to the next measurent.
            measurement = next(count_iterator)

            # Constructing propositional solution.
            solution = []
            for i in range(1,len(literals)+1):

                # Converting the measurent string to solution array.
                solution.append(literals[i-1]) if bool(int(measurement[-i])) else solution.append(-literals[i-1])

            # Return the proposed solution if correct.
            if sat.verify(cnf, solution) and compare_string == len(compare_string)*['1']: return solution

            # Return the proposed solution if the number of correct clause is equal to the value of the compare_string.
            if sat.verify_max(cnf, solution, compare_string) and compare_string != len(compare_string)*['1']: return solution

            # Return None if no solution was found.
            if m >= sqrt(2^len(literal_qubits)): return None

            # Increment iterations upper bound.
            m = min(lamb*m , sqrt(2^len(literal_qubits)))

# Grover max-sat solver function with adder oracle.
def grover_sat_solver_max(cnf, number, lamb, shots):

    # Get number of bits for compare_string.
    number_bits = ceil(log(sum(number))/log(2))

    # Define compare string.
    compare_string = number_bits*['x']

    # Find the maximum total weight.
    for i in range(number_bits):

        # Set compare_string to 1 and look at the result, otherwise set to 0.
        compare_string[number_bits-1-i] = '1'

        # Get intermediate result.
        intermediate_result = grover_sat_solver_adder(cnf, number, compare_string, lamb, shots)
        if intermediate_result == None: compare_string[number_bits-1-i] = '0'

        # Update result if successful.
        if intermediate_result != None: result = intermediate_result

    return result, compare_string
