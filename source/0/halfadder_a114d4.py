# https://github.com/Wirkungsfunktional/QuantumInformationProcessing/blob/b3929f9a8e838e3dbda9feee5cf753d1204e75a1/QuantumGates/HalfAdder.py
from qiskit import QuantumProgram



qp = QuantumProgram()
qr = qp.create_quantum_register('qr',4)
cr = qp.create_classical_register('cr',4)
qc = qp.create_circuit('Half_Adder',[qr],[cr])

inp = input("val: ")


if inp == "00":
    pass
elif inp == "01":
    qc.x(qr[3])
elif inp == "10":
    qc.x(qr[2])
elif inp == "11":
    qc.x(qr[2])
    qc.x(qr[3])



qc.ccx(qr[3], qr[2], qr[1])
qc.cx(qr[3], qr[0])
qc.cx(qr[2], qr[0])


qc.measure(qr[0], cr[0])
qc.measure(qr[1], cr[1])
qc.measure(qr[2], cr[2])
qc.measure(qr[3], cr[3])

result = qp.execute('Half_Adder')
print(result.get_counts('Half_Adder'))
