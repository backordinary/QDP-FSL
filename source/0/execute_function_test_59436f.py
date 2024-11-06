# https://github.com/dmiracle/qiskit-test/blob/f7c570b6a52cddbbcadacb317e89125d13276dc1/execute_function_test.py
import qiskit
# from qiskit import Aer
# qasm_str = """OPENQASM 2.0;
#         include "qelib1.inc";
#         qreg q[5];
#         creg c[5];
#         x q[4];
#         h q[3];
#         h q[4];
#         cx q[3],q[4];
#         h q[3];
#         measure q[3] -> c[3];"""

# qc = qiskit.QuantumCircuit.from_qasm_str(qasm_str)

qc = qiskit.QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
# backend = Aer.get_backend("qasm_simulator")
# result = qiskit.execute(qc, backend, shots=50)
drawing = qc.draw(output='mpl', scale=0.7)
drawing.savefig('test.svg', format="svg", bbox_inches="tight", transparent=True)
