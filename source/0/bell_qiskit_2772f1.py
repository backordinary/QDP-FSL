# https://github.com/dmiracle/ics-698/blob/24840c89d20976be998e4c158ec02885f8d51741/final-report/bell-qiskit.py
#import strangeworks.qiskit to use formatted result data and visualizations
import qiskit

qc = qiskit.QuantumCircuit.from_qasm_file("bell.qasm")
qc.h(0)
# qc.cx(0, 1)

#don't forget to use the strangeworks backend to access visualizations
simulator = qiskit.get_backend('statevector_simulator')

result = qiskit.execute(qc, simulator).result()