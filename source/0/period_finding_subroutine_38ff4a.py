# https://github.com/StijnW66/Quantum-Project/blob/a50d0c76af13194c139d7302cbb5f64dc7c659e6/src/quantum/period_finding_subroutine/period_finding_subroutine.py
import sys
sys.path.append(".")

from fractions import Fraction

from qiskit import transpile, assemble, Aer, QuantumRegister, ClassicalRegister, QuantumCircuit

from src.quantum.gates.control_qubits import control_qubits
from src.quantum.gates.one_control_qubit import classic_one_control_qubit
from src.quantum.qi_runner import execute_circuit



experiment_number_of_shots = 1024
quantum_subroutine_attempt_number = 0


def find_period(a, N):
    size = len(bin(N)) - 2

    c = QuantumRegister(2 * size)
    q = QuantumRegister(2 * size + 2)
    clas = ClassicalRegister(2 * size)
    circuit = QuantumCircuit(c, q, clas)

    circuit.append(control_qubits(size, a, N), c[:] + q[:])
    circuit.measure(range(2 * size), range(2 * size))
    for i in range(100):
        print("trying to find r. iteration:", i)
        qi_result = execute_circuit(circuit, 1)

        counts_histogram = qi_result.get_counts(circuit)
        bin_result = counts_histogram.most_frequent()[2 + size: 2 * size + 2]
        decimal = int(bin_result[::-1], 2)
        phase = decimal / (2 ** (2 * size))
        f = Fraction(phase).limit_denominator(N)
        r = f.denominator
        print(decimal, phase, r)
        if r > 2:
            if (a ** r) % N == 1:
                return r
    # too many tries
    return 1


def find_period_2n_plus_3(a, N):
    # Print message indicating start of period finding subroutine
    global quantum_subroutine_attempt_number
    quantum_subroutine_attempt_number += 1
    print("\tPeriod Finding Routine: Attempt " + str(quantum_subroutine_attempt_number))

    # Generate circuit for finding based on passed values
    size = len(bin(N).lstrip("0b"))
    circuit = classic_one_control_qubit(size, a, N)

    # Execute the circuit pn the Qiskit Aer simulator
    global experiment_number_of_shots
    aer_sim = Aer.get_backend('aer_simulator')
    t_qc = transpile(circuit, aer_sim)
    qobj = assemble(t_qc, shots=experiment_number_of_shots)
    results = aer_sim.run(qobj).result()
    counts = results.get_counts(circuit)
    print("\t\tExperiment of " + str(experiment_number_of_shots) + " shots finished.")

    # Process experiment results
    n_count = 2 * size
    rows = []
    for output in counts:
        decimal = int(output[::-1], 2)  # Convert (base 2) string to decimal
        phase = decimal / (2 ** n_count)  # Find corresponding eigenvalue (phase)
        rows.append([counts[output], Fraction(phase).limit_denominator(N)])

    # Sort in descending order by counts
    rows.sort(reverse=True, key=lambda x: x[0])

    # Checking top 10 phases' estimations for smallest 'r' satisfying a^r mod N == 1
    r = sys.maxsize
    guess_no = 0
    for idx, row in enumerate(rows[0:9]):
        if a ** row[1].denominator % N == 1 and row[1].denominator < r:
            r = row[1].denominator
            guess_no = idx + 1

    # Print message if subroutine found a period 'r' of a^r mod N == 1
    if r != sys.maxsize:
        print(f"\t\tPeriod finding routine successful. Phase estimation number {guess_no} yielded correct answer.")
        quantum_subroutine_attempt_number = 0
    else:
        print("\t\tPeriod finding routine unsuccessful. Repeating the experiment.")

    # Return obtained 'r'
    return r
