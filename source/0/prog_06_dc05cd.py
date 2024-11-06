# https://github.com/1chooo/Programming-Evolution/blob/ab8c8e388ab098163eebd736d47fe9a559ad1090/NCU/sophomore/CE3005/alg/quantum/CH01/prog_06.py
from qiskit import QuantumCircuit, IBMQ, execute 
from qiskit.tools.monitor import job_monitor

qc = QuantumCircuit(1, 1)
qc.measure([0], [0])
print(qc)
IBMQ.save_account('your_token')
IBMQ.load_account()
IBMQ.providers()
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
qcomp = provider.get_backend('ibmq_manila')
job = execute(qc, backend = qcomp, shots = 1000)
job_monitor(job)
result = job.result()
counts = result.get_counts(qc)
print("Total counts for qubit states are:", counts)