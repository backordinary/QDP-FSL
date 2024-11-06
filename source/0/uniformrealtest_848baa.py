# https://github.com/StoianAlinBogdan/ProiectLicenta/blob/0b7b902b615ed6030de93a17dab6f8f0e91934ba/UniformRealTest.py
from qiskit import IBMQ, assemble, transpile
from qiskit_finance.circuit.library import UniformDistribution

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')

backend = provider.get_backend('ibmq_santiago')

qc = UniformDistribution(6)
qc.measure_all()
mapped_qc = transpile(qc, backend=backend)
qobj = assemble(mapped_qc, backend=backend, shots=1024)
job = backend.run(qobj)

result = job.result()
counts = result.get_counts()
print(counts)
