# https://github.com/cristiannaziru/QOSF-Playground/blob/bb3a551219aa26f8c6da6c986554fde38cd5363d/multiplier.py
from qiskit import Aer, execute
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import RGQFTMultiplier

def multiplier(a, b):
  # Turning the inputs into binary strings
  a_binary = format(a, '02b')
  b_binary = format(b, '02b')
  result_binary = format(a*b, '02b')

  # Appending zeros to the binary numbers to make them the same length
  if len(a_binary) > len(b_binary):
      b_binary = '0'*(len(a_binary)-len(b_binary)) + b_binary
  elif len(b_binary) > len(a_binary):
      a_binary = '0'*(len(b_binary)-len(a_binary)) + a_binary

  # Creating the registers
  quantum_register = QuantumRegister(len(a_binary) + len(b_binary) + len(result_binary), 'q')
  classical_register = ClassicalRegister(len(result_binary), 'c')

  # Creating the circuit
  circuit = QuantumCircuit(quantum_register, classical_register)

  # Initializing the registers
  for i in range(len(a_binary)):
    if a_binary[i] == '1':
      circuit.x(quantum_register[i])

  for i in range(len(b_binary)):
    if b_binary[i] == '1':
      circuit.x(quantum_register[i+len(a_binary)])

  # Creating the multiplier
  multiplier = RGQFTMultiplier(num_state_qubits=len(a_binary), num_result_qubits=len(result_binary))

  # Appending the multiplier to the circuit
  circuit.append(multiplier, quantum_register[:])

  # Adding measurements
  circuit.measure(quantum_register[len(a_binary)+len(b_binary):], classical_register)

  # Print the results
  circuit.draw(output='mpl')

  # Execute the circuit on the backend

  backend = Aer.get_backend('qasm_simulator')
  job = execute(circuit, backend, shots=1)
  result = job.result()
  counts = result.get_counts(circuit)
  print(counts)
  return counts
