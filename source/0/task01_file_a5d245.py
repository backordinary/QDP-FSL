# https://github.com/talashilkarraj/QOSF_Task1/blob/e2ff3e29aa8e46e5b9c9197189f32d70f1801e69/Task01_file.py
import numpy as np
from matplotlib import pyplot as plt

# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, Aer, IBMQ,  assemble
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit.visualization import plot_state_qsphere, plot_state_city, plot_histogram


"""
Function: Decimal to Binary Conversion
Parameters: Number to be converted
Returns: Decimal to binary converted String 
"""
def decimal_To_binary(num):
  binary = str(bin(num))
  return binary[2:len(binary)]


"""
Function: Largest_Number
Parameters: List
Returns: Tuple wtih 2 values - largest number in the list, length of binary converted 
         largest number
"""
def largest_number(list):
  big = list[0]
  for i in range(0,len(list)-1):
    if big>list[i+1]:
      pass
    else: 
      big=list[i+1]
  return big,len(decimal_To_binary(big))


"""
Function: Binary converted list 
Parameters: List
Returns: a list with every number of the list converted into its equivalent binary 
         with bitlength same for all numbers & equal to the bitlength of the largest number
"""
def binary_converted_list(list):
  big, max_bit_length = largest_number(list)
  list_of_binaries=[]
  for i in list:
    k = decimal_To_binary(i)
    if (len(k) == max_bit_length):
      pass
    else: 
      k = "0"*(max_bit_length-len(k))+k
    list_of_binaries.append(k)
  return(list_of_binaries)


"""
Function: Binary alternate digits list
Parameters: List
Returns: tuple of 2 lists: 1st list of binary indices of the elements having alternate 0s and 1s
        2nd list of elements having alternate 0s & 1s.
"""
def binary_alternate_digits_list(list):
  list_of_alternate_digits=[]
  indices=[]
  print(list)
  counter = 0
  for i in list:
    
    for j in range(0,len(i)-2):
      if int(i[j]) != int(i[j+2]):
        break
    for j in range(0,len(i)-1):
      if int(i[j]) == int(i[j+1]):
          break
    else:
      list_of_alternate_digits.append(i)
      indices.append((counter))
    counter+=1
  
  indices = binary_converted_list(indices)
  return indices, list_of_alternate_digits


totalQubits = 2 
state00 = '00'
state01 = '01'
state11 = '11'
state10 = '10'
vector = [1,5,7,10]

list = binary_alternate_digits_list(binary_converted_list(vector))
states = list[0]                                         # List containing states to be linearly combined    
print("States to be linearly combiined: "+ str(states))

ckt = QuantumCircuit(totalQubits,totalQubits)            # Main circuit to be converted into grover's


"""
Function: Diffuser nqubits
Parameters: circuit
Returns: diffuser gate applicable to N qubits, to be applied after oracle to amplify
         the required output sate
"""
def grover_circuit_2qubits(ckt,states):
    ckt.h([0,1])

    if state00 in states and state01 in states:
        oracle_2qubits(state00, state01, ckt)
        diffuser_2qubits(state00, state01, ckt)

    if state01 in states and state11 in states:
        oracle_2qubits(state01, state11, ckt)
        diffuser_2qubits(state01, state11, ckt) 
    
    if state11 in states and state10 in states:
        oracle_2qubits(state11, state10, ckt)
        diffuser_2qubits(state11, state10, ckt)
    
    if state10 in states and state00 in states:
        oracle_2qubits(state10, state00, ckt)
        diffuser_2qubits(state10, state00, ckt)
    
    if state00 in states and state11 in states:
        oracle_2qubits(state00, state11, ckt)
        diffuser_2qubits(state00, state11, ckt)

    if state10 in states and state01 in states:
        oracle_2qubits(state10, state01, ckt)
        diffuser_2qubits(state10, state01, ckt)

    # universal diffuser for qubits>3
    # if totalQubits>2:
    #     for i in range(0,5):
    #         ckt.append(diffuser_nqubits(totalQubits),[0,1])
    ckt.measure_all()

    # qubits = list(range(totalQubits))     // for debugging and checking
    # print(qubits)                         // for debugging and checking
    ckt.draw(output='mpl')


"""
Function: Diffuser 2qubits
Parameters: circuit
Returns: No return parameter
"""
def diffuser_2qubits(stateAA,stateBB,circuit):
    if stateAA is state00 and stateBB is state01:
        circuit.h([0,1])
        circuit.x([0,1])
        circuit.h(1)
        circuit.x([0,1])
        circuit.h([0,1])
    if stateAA is state01 and stateBB is state11:
        circuit.h([0,1])
        circuit.x([0,1])
        circuit.cx(1,0)
        circuit.h(0)
        circuit.x([0,1])
        circuit.h([0,1])
    if stateAA is state11 and stateBB is state10:
        circuit.h([0,1])
        circuit.x([0,1])
        circuit.h(1)
        circuit.x(1)
        circuit.x([0,1])
        circuit.h([0,1])
    if stateAA is state10 and stateBB is state00:
        circuit.h([0,1])
        circuit.x([0,1])
        circuit.cx(0,1)
        circuit.h(0) 
        circuit.x([0,1])
        circuit.h([0,1])
    if stateAA is state00 and stateBB is state11:
        circuit.h([0,1])
        circuit.x([0,1])
        circuit.h(0) 
        circuit.cx(0,1)
        circuit.x([0,1])
        circuit.h([0,1])
    if stateAA is state10 and stateBB is state01:
        circuit.h([0,1])
        circuit.x([0,1])
        circuit.x(0)
        circuit.h(0)
        circuit.cx(0,1)
        circuit.x([0,1])
        circuit.h([0,1])


"""
Function: oracle 2qubits
Parameters: circuit
Returns: No return parameter
"""
def oracle_2qubits(stateAA,stateBB,circuit):
    if stateAA is state00 and stateBB is state01:
        circuit.z(1)
        circuit.x(1)
    if stateAA is state01 and stateBB is state11:
        circuit.z(0)
    if stateAA is state11 and stateBB is state10:
        circuit.z(1)
    if stateAA is state10 and stateBB is state00:
        circuit.z(0)
        circuit.x(0)
    if stateAA is state00 and stateBB is state11:
        circuit.z([0,1])
        circuit.x(1)
    if stateAA is state10 and stateBB is state01:
        circuit.z([0,1])


"""
Function: Diffuser nqubits
Parameters: circuit
Returns: diffuser gate applicable to N qubits, to be applied after oracle to amplify
         the required output sate
"""
def diffuser_nqubits(totalQubits):
  dckt = QuantumCircuit(totalQubits)
  for qubit in range(0,totalQubits):
    dckt.h(qubit)
  
  for qubit in range(0,totalQubits):
    dckt.x(qubit)

  dckt.h(totalQubits-1)
  if totalQubits==2:
    dckt.cx(0,1)
  else:
    dckt.mct(list(range(0,totalQubits-1)),totalQubits-1)
  dckt.h(totalQubits-1)

  for qubit in range(0,totalQubits):
    dckt.x(qubit)

  for qubit in range(0,totalQubits):
    dckt.h(qubit)

  #dckt.draw(output='mpl')  //for debugging and checking
  return dckt

#Calling Main grover circuit
grover_circuit_2qubits(ckt,states)


#Uncomment for simulation with statevector simulator
"""
Simulating with statevector simulator
"""
# backend = Aer.get_backend('statevector_simulator')
# job = backend.run(ckt)
# result = job.result()
# outputstate = result.get_statevector()
# print("Output State: "+ str(outputstate))
# state2 = Statevector.from_instruction(ckt)
# x = plot_state_qsphere(state2)

#Uncomment for simulations with aer simulator
"""
Using aer simulator
"""
aer_sim = Aer.get_backend('aer_simulator')
transpiled_grover_circuit = transpile(ckt, aer_sim,optimization_level=3)
qobj = assemble(transpiled_grover_circuit)
result = aer_sim.run(qobj).result()
counts = result.get_counts()
plot_histogram(counts)

plt.show()
