# https://github.com/stoicswe/CSCI395A-QuantumComputing/blob/7bd82873d4477ca0b4daab96964ebc44b9036671/quantumWalk_version1/quantumWalkStart1.py
from qiskit import QuantumProgram

Q_Program = QuantumProgram()

#initialize the registers
qr = Q_Program.create_quantum_register("qr", 5)
cr = Q_Program.create_classical_register("cr", 5)
qc = Q_Program.create_circuit("quantumWalk", [qr], [cr])

#step 1
qc.x(qr[0])
qc.h(qr[0])
qc.ccx(qr[4], qr[3], qr[0])
qc.ccx(qr[0], qr[2], qr[1])
qc.ccx(qr[4], qr[3], qr[0])

qc.ccx(qr[0], qr[1], qr[2])
qc.cx(qr[0], qr[1])
qc.x(qr[0])
qc.measure(qr[1], cr[1])
qc.measure(qr[2], cr[2])
qc.measure(qr[3], cr[3])
qc.measure(qr[4], cr[4])

#step 2
qc.h(qr[0])
qc.ccx(qr[4], qr[3], qr[0])
qc.ccx(qr[0], qr[2], qr[1])
qc.ccx(qr[4], qr[3], qr[0])

qc.ccx(qr[0], qr[1], qr[2])
qc.cx(qr[0], qr[1])
qc.x(qr[0])
qc.measure(qr[1], cr[1])
qc.measure(qr[2], cr[2])
qc.measure(qr[3], cr[3])
qc.measure(qr[4], cr[4])

#step 3
qc.h(qr[0])
qc.ccx(qr[4], qr[3], qr[0])
qc.ccx(qr[0], qr[2], qr[1])
qc.ccx(qr[4], qr[3], qr[0])

qc.ccx(qr[0], qr[1], qr[2])
qc.cx(qr[0], qr[1])
qc.x(qr[0])
qc.measure(qr[1], cr[1])
qc.measure(qr[2], cr[2])
qc.measure(qr[3], cr[3])
qc.measure(qr[4], cr[4])

#step 4
qc.h(qr[0])
qc.ccx(qr[4], qr[3], qr[0])
qc.ccx(qr[0], qr[2], qr[1])
qc.ccx(qr[4], qr[3], qr[0])

qc.ccx(qr[0], qr[1], qr[2])
qc.cx(qr[0], qr[1])
qc.x(qr[0])
qc.measure(qr[1], cr[1])
qc.measure(qr[2], cr[2])
qc.measure(qr[3], cr[3])
qc.measure(qr[4], cr[4])

#run the complete circuit and gather the result
result = Q_Program.execute(["quantumWalk"], backend='local_qasm_simulator', shots=1024)

#return the result to display
print(result)
print(result.get_data("quantumWalk"))