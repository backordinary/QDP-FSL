# https://github.com/hflash/profiling/blob/0498dff8c1901591d4428c3149eb3aadf80ac483/testfiles/test_program.py
from qiskit import QuantumCircuit, IBMQ, execute
from qiskit.visualization import plot_histogram
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor

filename = '../examples/bv_n5.qasm'
qc = QuantumCircuit.from_qasm_file(filename)
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
backend = least_busy(provider.backends(filters=lambda b:
                                        b.configuration().n_qubits >= 3 and
                                        not b.configuration().simulator and
                                        b.status().operational == True))
job_exp = execute(qc, backend=backend, shots=8192)
job_monitor(job_exp, interval=2)
exp_result = job_exp.result()
exp_measurement_result = exp_result.get_counts()
plot_histogram(exp_measurement_result).savefig('real_result.png')
