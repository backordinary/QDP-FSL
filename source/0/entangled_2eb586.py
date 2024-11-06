# https://github.com/bchatterjee99/quantum/blob/7137eddb16ec3916fd7211f9cd3c62c47aa54e7a/entangled.py
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, Aer, execute

q1 = QuantumRegister(2);
c1 = ClassicalRegister(2);

qc = QuantumCircuit(q1, c1);
backend = Aer.get_backend("qasm_simulator");


#-----------------------------------------------------------
#
#       +---------+                                         
#       |         |                                         
# |0> --|    H    |------------------------#------
#       |         |   |0>+|1>              |               
#       +---------+  ---------             |     
#                     sqrt(2)              |   |01> + |10>    
#                                          |   ----------- 
#                                          |     sqrt(2)   
#       +---------+                        |    
#       |         |                        |    
# |0> --|    X    |-----------------------|+|-----
#       |         |     |1>                    
#       +---------+              |                   
#                                |                    
#                                v                  
#                                                              
#                            |01> + |11>                     
#       		     -----------
#                              sqrt(2)              
#
#-----------------------------------------------------------

qc.h(q1[0]);
qc.x(q1[1]);
qc.cx(q1[0], q1[1])
qc.measure(q1, c1);

qjob = execute(qc, backend, shots=100);
counts = qjob.result().get_counts();

print(counts)
