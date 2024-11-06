# https://github.com/shikharsingh3/Qosf_Screening_Task_2/blob/3572f73b12f0756b3c18fd0ccd833f24327baae3/.python/Task_2.py
""" Quantum Mentorship Program -- QOSF -- Screening Task """

from qiskit import(
  QuantumCircuit, QuantumRegister, ClassicalRegister,
  execute,
  Aer)

"""Task2"""

"""The Bell state (|00〉 + |11〉)/sqrt(2), which is a maximally entangled state, is created by initialising
a two qubit input state to |00〉 and applying a Hadamard gate to the first qubit and a CNOT gate with 
first qubit as control input and second qubit as target input. In this task, an error gate is added to
each of the two qubits before the CNOT gate. The error can be a bit-flip error (X gate), sign-flip 
error (Z gate) or an identity gate. Each of the two qubits are encoded using bit-flip or sign-flip code
and are subsequently measured in the Z-basis. On measuring the first qubit for the state (|00〉 + |11〉)/sqrt(2), 
the result of measuring the second qubit is guaranteed to yield the same value - '00' or '11'.

The first qubit state |0〉 is encoded using only sign-flip code because after applying Hadamard gate, a sign-flip error
would flip |+〉 to |-〉 but a bit-flip error would leave the state |+〉 unchanged. Similarly, the second qubit state |0〉 
is encoded using only bit-flip code because a sign-flip error would leave the state |0〉 unchanged. This eliminates the 
implementation of Shor Code to both of the input qubit states. Hence, the problem is simplified to encoding the 
first qubit state |+〉 as |+++〉 and the second qubit state |0〉 as |000〉. Now since the encoded states are only |+++〉 
and |000〉, the application of CNOTs during encoding for the sign-flip and bit-flip code is not required.
"""

def encoder(register):
 """
 Encodes into three-qubit phase-flip code by applying Hadamard gate to first three qubits of the 6 qubit register.
 Also used to recover the state in |0〉, |1〉 basis during recovery by applying inverse of Hadamard gate (also the Hadamard gate).

 Args:
  register: The 6 qubit register to be encoded or recovered.
 """
 register.h(0)
 register.h(1)
 register.h(2)

def decoder(register, dQ):
 """
 Recovers the qubit state (after applying Hadamard gate to each of the qubits-Encoder()) 
 using Clifford + T gate set.

 Args:
  register: The 6 qubit register to be decoded.
  dQ: The increment for the three-qubit register. 
 """
 register.cx(dQ, 1 + dQ)
 register.cx(dQ, 2 + dQ)
 register.ccx(1 + dQ, 2 + dQ, dQ)

def BellStateCode(errorGate1, errorGate2):
 """
 For error gates on the 6 qubit register, the function returns the measurement result of the Bell State. The first and fourth
 qubits of the register are measured.

 Args:
  errorGate1: Error gate applied on the first 3 qubits of the 6 qubit register. 
  Bit-flip error (X gate), sign-flip error (Z gate) or Identity gate.
  errorGate2: Error gate applied on the last 3 qubits of the 6 qubit register.
  Bit-flip error (X gate), sign-flip error (Z gate) or Identity gate.

 Returns:
  List of measurement outcomes stored in memory.
 """
 qreg = QuantumRegister(6)
 creg = ClassicalRegister(2)
 circ = QuantumCircuit(qreg, creg)

 """Encoding"""
 encoder(circ)

 """Error"""
 circ.append(errorGate1, [qreg[0], qreg[1], qreg[2]])
 circ.append(errorGate2, [qreg[3], qreg[4], qreg[5]])

 """Recovery"""
 encoder(circ)
 decoder(circ, 0)
 decoder(circ, 3)
  
 circ.h(0)
 circ.cx(0, 3)

 """Bell state measurement in Z-basis"""
 circ.measure([0,3], [0,1])
 backend = Aer.get_backend('qasm_simulator')
 counts = execute(circ, backend, shots = 1000).result().get_counts(circ)
 memory = execute(circ, backend, shots=1000, memory=True).result().get_memory(circ)
 print(counts)
 return memory