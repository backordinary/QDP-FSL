# https://github.com/qiskit-community/qiskit-braket-provider/blob/66de72ba607e786229fd9964a77a8de6acb3814e/docs/how_tos/data/3_hybrid_jobs/algorithm_script.py
"""Example of usage of Qiskit-Braket provider."""
from qiskit import QuantumCircuit
from qiskit_braket_provider import AWSBraketProvider

from braket.jobs import save_job_result


provider = AWSBraketProvider()
backend = provider.get_backend("SV1")
circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)

results = backend.run(circuit, shots=1)

print(results.result().get_counts())
save_job_result(results.result().get_counts())
