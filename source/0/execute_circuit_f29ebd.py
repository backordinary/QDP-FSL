# https://github.com/hflash/profiling/blob/0498dff8c1901591d4428c3149eb3aadf80ac483/circuittransform/execute_circuit.py
from qiskit import execute, QuantumCircuit, IBMQ, Aer

filename = "./qasm/qft_10.qasm"
circuit = QuantumCircuit.from_qasm_file(filename)
circuit.measure_all()
IBMQ.load_account()
backend = Aer.get_backend()
execute()
