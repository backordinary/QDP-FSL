# https://github.com/psasanka1729/Grover-NISQ/blob/a410b3273db970396ab0b9ac31cd0cd5b0b1eec4/two%20level%20gates/ClassiQ_mcx.py
#!/usr/bin/env python
# coding: utf-8

'''

Decompose a MCX gate with 14 control qubits and a maximum of
5 ancilla qubits into single and double qubits CX gates with a circuit of minimal depth.

Author - Sasanka Dowarah, Saeed Rahmanian Koshkaki, Michael H. Kolodrubetz.

'''


from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile


''' Setting up the circuit '''

qr = QuantumRegister(14, 'c') ## 14 control qubits.
target_qubit = QuantumRegister(1,'t') ## target qubit.
anc = QuantumRegister(5, 'ancilla') ## 5 ancilla qubits.
qc = QuantumCircuit(qr,target_qubit, anc)
qc.mcx(qr[3*0:3*0+3], anc[0])
qc.mcx(qr[3*1:3*1+3], anc[1])
qc.mcx(qr[3*2:3*2+3], anc[2])
qc.mcx(qr[3*3:3*3+3], anc[3])
qc.mcx([qr[12],qr[13],anc[0]],anc[4])

## 4-qubit gate.
qc.mcx(anc[1:5],target_qubit[0])

## Apply the 3-qubit gates in reverse to return controls and ancillas to original state.
qc.mcx([qr[12],qr[13],anc[0]],anc[4])
for i in range(3,-1,-1):
    qc.mcx(qr[3*i:3*i+3], anc[i])



''' Uncomment the following line to save the circuit as png. '''

#qc.draw('mpl') #.savefig('3_4_breakdown.png', dpi=600)



## Decompose the above circuit consisting of 3 and 4 qubit gates in the basis {U,CX}.
circuit_decomposition= transpile(qc, basis_gates=['u','cx'], optimization_level = 3)




''' Uncomment the following line to save the final transpiled circuit as png. '''

#circuit_decomposition.draw('mpl') #.savefig('transpiled.png',dpi = 300)


print("Circuit depth =", circuit_decomposition.depth())

print("Number of 1 and 2 qubit gates =",circuit_decomposition.count_ops())