# https://github.com/StijnW66/Quantum-Project/blob/f9b6abb906e89f70e665dd53b408e07fabab4b7d/src/quantum/qi_runner.py
import os
from quantuminspire.credentials import get_authentication
from quantuminspire.qiskit import QI
from qiskit import execute
from qiskit import QuantumCircuit, Aer, transpile, assemble



qi_backend = None

# Method that sets up the quantuminspire api with given project_name and backend_name.
def setup_QI(project_name="Shor's algorithm", backend_name='QX single-node simulator'):
    global qi_backend
    QI_URL = os.getenv('API_URL', 'https://api.quantum-inspire.com/')
    authentication = get_authentication()
    QI.set_authentication(authentication, QI_URL, project_name=project_name)
    qi_backend = QI.get_backend(backend_name)

# Method that executes a given circuit.
def execute_circuit(circuit, shots=256):
    if qi_backend is None:
        print('Running on local AER simulator')
        aer_sim = Aer.get_backend('aer_simulator')
        t_qc = transpile(circuit, aer_sim)
        #qobj = assemble(t_qc)
        results = aer_sim.run(t_qc).result()
        print(results)
        return results

    qi_job = execute(circuit, backend=qi_backend, shots=shots)
    qi_result = qi_job.result()

    return qi_result

# Method that is able to print the results retrived from the execute_circuit method.
def print_results(qi_result, circuit):
    counts_histogram = qi_result.get_counts(circuit)
    print('\nState\tCounts')
    [print('{0}\t\t{1}'.format(state, counts)) for state, counts in counts_histogram.items()]

    # Print the full state probabilities histogram
    probabilities_histogram = qi_result.get_probabilities(circuit)
    print('\nState\tProbabilities')
    [print('{0}\t\t{1}'.format(state, val)) for state, val in probabilities_histogram.items()]