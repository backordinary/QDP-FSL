# https://github.com/UST-QuAntiL/code-injection-example/blob/e7997c42fb849cba2d49dc508b74702dcb1643e6/user_code/test_package/user_code/external.py
import qiskit
from qiskit import execute as renamed_execute, QuantumCircuit


def extern_func():
	qc = QuantumCircuit(1)
	qc.h(0)
	qc.measure_all()

	backend = qiskit.Aer.get_backend("qasm_simulator")
	job = renamed_execute(qc, backend)

	result = job.result().get_counts(qc)
	print(result)
	return result
