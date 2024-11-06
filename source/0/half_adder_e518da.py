# https://github.com/parasol4791/quantumComp/blob/8b52dba9a3f8cbddf3b1f29ef3500ca00a840b2e/algos/half_adder.py
# https://qiskit.org/textbook/ch-states/atoms-computation.html
# This algo is for adding 2 bits
# 0+0 = 00
# 0+1 = 01
# 1+0 = 01
# 1+1 = 10
# The right bit (qubit 2) is 0 when both control bits (qubits 0 and 1) are the same, and 1 otherwise. It's implemented by 2 CNOT gates.
# The left bit (qubit 3) is 1 only when both control bits (qubits 0 and 1) are 1, and 0 otherwise. It's implemented by CCNOT (Toffoli) gate.

from qiskit import QuantumCircuit, Aer

qc_ha = QuantumCircuit(4,2)
# encode inputs in qubits 0 and 1
qc_ha.x(0) # For a=0, remove the this line. For a=1, leave it.
qc_ha.x(1) # For b=0, remove the this line. For b=1, leave it.
qc_ha.barrier()
# use cnots to write the XOR of the inputs on qubit 2
qc_ha.cx(0,2)
qc_ha.cx(1,2)
# use ccx to write the AND of the inputs on qubit 3
qc_ha.ccx(0,1,3)
qc_ha.barrier()
# extract outputs
qc_ha.measure(2,0) # extract XOR value
qc_ha.measure(3,1) # extract AND value

qc_ha.draw()
#      ┌───┐ ░                 ░
# q_0: ┤ X ├─░───■─────────■───░───────
#      ├───┤ ░   │         │   ░
# q_1: ┤ X ├─░───┼────■────■───░───────
#      └───┘ ░ ┌─┴─┐┌─┴─┐  │   ░ ┌─┐
# q_2: ──────░─┤ X ├┤ X ├──┼───░─┤M├───
#            ░ └───┘└───┘┌─┴─┐ ░ └╥┘┌─┐
# q_3: ──────░───────────┤ X ├─░──╫─┤M├
#            ░           └───┘ ░  ║ └╥┘
# c: 2/═══════════════════════════╩══╩═
#                                 0  1

sim = Aer.get_backend('aer_simulator')
counts = sim.run(qc_ha).result().get_counts()
print(counts)
# (qubits 0,1)
