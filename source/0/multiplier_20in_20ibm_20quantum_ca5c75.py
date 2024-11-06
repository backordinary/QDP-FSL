# https://github.com/Banani456/QOSF_Screening_Task_C6_Multiplier/blob/2d6089c3eb8ebd146c96aa33582117cc728482a6/Multiplier%20in%20IBM%20Quantum.py
"""
Can multiply up to 15*15
"""

from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, execute,IBMQ
from qiskit.tools.monitor import job_monitor
from qiskit.circuit.library import RGQFTMultiplier
from math import*

IBMQ.enable_account('Your API Key HERE')
provider = IBMQ.get_provider(hub='ibm-q')

backend = provider.get_backend('ibmq_qasm_simulator')

q = QuantumRegister(16,'q')
c = ClassicalRegister(8,'c')

circuit = QuantumCircuit(q,c)

"""
converts to number to binary
"""
list = []
def ctb(num, base): 
  converted_string, mod = "", num % base 
 
  while num != 0: 
    mod = num % base
    list.append(mod)
    num = int(num / base) 

  list.reverse()
  return list

"""
done
"""

"""
multiplier time
"""
def multiplier(num1, num2):

#first operand
    qnum1 = ctb(num1,2)
    if qnum1[0] == 1:
        circuit.x(q[0])
    if qnum1[1] == 1:
        circuit.x(q[1])
    if qnum1[2] == 1:
        circuit.x(q[2])
    if qnum1[3] == 1:
        circuit.x(q[3])
    
#second operand
    qnum2 = ctb(num2,2)
    if qnum2[0] == 1:
        circuit.x(q[4])
    if qnum2[1] == 1:
        circuit.x(q[5])
    if qnum2[2] == 1:
        circuit.x(q[6])
    if qnum2[3] == 1:
        circuit.x(q[7])

    circuit1 = RGQFTMultiplier(num_state_qubits=4, num_result_qubits=8)
    circuit = circuit.compose(circuit1)

    circuit.measure(q[8],c[0])
    circuit.measure(q[9],c[1])
    circuit.measure(q[10],c[2])
    circuit.measure(q[11],c[3])
    circuit.measure(q[12],c[4])
    circuit.measure(q[13],c[5])
    circuit.measure(q[14],c[6])
    circuit.measure(q[16],c[7])

    print(circuit)

    job = execute(circuit, backend, shots=2000)
    result = job.result()
    counts = result.get_counts()

number1 = input("Enter number between 1-15:")
number2 = input("Enter another number between 1-15:")
print(number1+'*'+number2)
print(multiplier(number1,number2))