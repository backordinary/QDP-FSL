# https://github.com/chaniotakismg/quantum-experiments/blob/5490ddd16b6f36d6b5a9f6773a029e4421f7d092/basic/Bernstein-Vazirani.py
from qiskit import *
from qiskit.tools import job_monitor

if __name__ == '__main__':

    # Configuration
    secret_number = '10110111'
    secret_number_length = len(secret_number)
    qubit_series = range(secret_number_length)

    # Circuit setup
    circuit = QuantumCircuit(secret_number_length + 1, secret_number_length)
    circuit.h(qubit_series)
    circuit.x(secret_number_length)
    circuit.h(secret_number_length)
    circuit.barrier()
    # Check positions of 1 and apply a CNOT(CX) gate
    for index, check in enumerate(reversed(secret_number)):
        if check == '1':
            circuit.cx(index, secret_number_length)
    circuit.barrier()
    circuit.h(qubit_series)
    circuit.barrier()
    circuit.measure(qubit_series, qubit_series)
    circuit.draw(output='mpl')

    # Measurement
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    device = provider.get_backend('ibmq_16_melbourne')  # We need many qubits
    job = execute(circuit, backend=device, shots=1)
    print("Job ID in ibmq_16_melbourne:", job.job_id())
    job_monitor(job)
    result = job.result()
    counts = result.get_counts()
    print("Guess and in iterations:", counts)
