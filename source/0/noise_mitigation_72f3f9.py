# https://github.com/chaniotakismg/quantum-experiments/blob/5490ddd16b6f36d6b5a9f6773a029e4421f7d092/basic/noise_mitigation.py
from ibm import register_token

from qiskit import *
from qiskit.providers.ibmq import IBMQ
from qiskit.visualization import plot_histogram
from qiskit.ignis.mitigation.measurement import (complete_meas_cal, CompleteMeasFitter)
from qiskit.tools.monitor import job_monitor

if __name__ == '__main__':
    register_token()

    qubits = 4

    circuit = QuantumCircuit(qubits, qubits)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.measure([0, 1, 2, 3], [0, 1, 2, 3])
    circuit.draw(output='mpl')

    simulator = Aer.get_backend('qasm_simulator')
    sim_result = execute(circuit, backend=simulator, shots=1024).result()

    plot_histogram(sim_result.get_counts(circuit))

    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q')
    device = provider.get_backend('ibmqx2')

    job = execute(circuit, backend=device, shots=1024)
    print("Job ID:", job.job_id())

    job_monitor(job)

    device_result = job.result()
    plot_histogram(device_result.get_counts(circuit))

    cal_circuits, state_labels = complete_meas_cal(qr=circuit.qregs[0], circlabel='measerrormitigationcal')
    cal_circuits[2].draw(output='mpl')