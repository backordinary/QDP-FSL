# https://github.com/asher-lab/Quantum-Computing-For-Humans/blob/5cb3924c7056cd8284db405fcb092a22ba4431e9/IBM-Quantum-Challenge/Week%201/week1a.py
#Create a full adder quantum circuit with input data:
# Where Inputs are A = 1, B = 0, X =0 

# If you run this code outside IBM Quantum Experience,
# run the following commands to store your API token locally.
# Please refer https://qiskit.org/documentation/install.html#access-ibm-quantum-systems
# IBMQ.save_account('MY_API_TOKEN')

# Loading your IBM Q account(s)
IBMQ.load_account()

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import IBMQ, Aer, execute

##### build your quantum circuit here
#Define registers and quantum unit
q = QuantumRegister(5)
c = ClassicalRegister(2)
qc = QuantumCircuit(q,c)
#NOT Gate
qc.x(q[0])
#CONTROL-NOT GATE
qc.cx(q[0],q[3])
qc.cx(q[1],q[3])
qc.cx(q[2],q[3])
#TOFOLLI GATE
qc.ccx(q[0],q[1],q[4])
qc.ccx(q[0],q[2],q[4])
qc.ccx(q[1],q[2],q[4])
#SUM OUT
qc.measure(q[3],c[0])
#Carry OUT
qc.measure(q[4],c[1])
######################B################
q = QuantumRegister(5,'q')
c = ClassicalRegister(2,'c')
qc = QuantumCircuit(q,c)
qc.x(q[1])
qc.cx(q[0],q[3])
qc.cx(q[1],q[3])
qc.cx(q[2],q[3])
qc.ccx(q[0],q[1],q[4])
qc.ccx(q[0],q[2],q[4])
qc.ccx(q[1],q[2],q[4])

qc.measure(q[3],c[0])
qc.measure(q[4],c[1])

##########X#############
q = QuantumRegister(5,'q')
c = ClassicalRegister(2,'c')

qc = QuantumCircuit(q,c)
qc.x(q[4])
qc.cx(q[0],q[3])
qc.cx(q[1],q[3])
qc.cx(q[2],q[3])
qc.ccx(q[0],q[1],q[4])
qc.ccx(q[0],q[2],q[4])
qc.ccx(q[1],q[2],q[4])

qc.measure(q[3],c[0])
qc.measure(q[4],c[1])



# execute the circuit by qasm_simulator
backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=1000)
result = job.result()
count =result.get_counts()
print(count)
qc.draw(output='mpl')
