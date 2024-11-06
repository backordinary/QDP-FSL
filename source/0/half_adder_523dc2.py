# https://github.com/Mostafa-Rawash/Quantum-Computing/blob/5107c2e97856886c069a73ce390a0dd76cf342ac/half-adder.py


from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator

qc2 = QuantumCircuit(4,2) # 4 qubit and  2 classical register // 4 to make processing , 2 to  print the result

qc2.x(0) 
qc2.x(1)
# .x() is quantum not-gate make both q0 and q1  = 1

qc2.cx(0,2)
qc2.cx(1,2)
# cx is XOR gate and save the return in sec. parameter if q0 != 0  return 0 to q2 then if q1 != 0 return 0 to q2  

qc2.ccx(0,1,3)
# ccx() in first to parameter = 1 will make  last one = 1 if one of them or both = 0 don't make any changes

# print the result in 2 classicical bits
qc2.measure([2,3],[0,1])
job = sim.run(qc2)
result= job.result()
result.get_counts()

# print the final circut  
qc2.draw()
