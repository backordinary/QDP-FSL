# https://github.com/wangpeihua1231/quantum_pharmaceutical_patent_search/blob/2915651e4a8be4ee91c192353b972b9952aded3a/4_qubit_noise.py
#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, Aer, IBMQ
import numpy as np
import math

from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import QuantumError, ReadoutError
from qiskit.providers.aer.noise import depolarizing_error
from qiskit.providers.aer.noise import thermal_relaxation_error

#Account Information
#provider = IBMQ.enable_account('your_account')


# In[3]:


def mark(bits, circuit, target):
    n = len(bits)
    texts = bits[::-1]
    ctrls = []
    for i in range(n):
        if texts[i] == "0":
            circuit.x(i)
            ctrls.append(i)
        elif texts[i] == "1":
            ctrls.append(i)
    circuit.ccx(0, 1, target)

    for i in range(n):
        if texts[i] == "0":
            circuit.x(i)
    qc.barrier()


# In[4]:


def code(texts1, texts2):
    output1, output2 = [], []
    for text in texts1:
        string = text.replace("C", "00")
        string = string.replace("(=O)", "01")
        string = string.replace("=O", "01")
        string = string.replace("O", "10")
        output1.append(string)
    for text in texts2:
        string = text.replace("C", "00")
        string = string.replace("(=O)", "01")
        string = string.replace("=O", "01")
        string = string.replace("O", "10")
        output2.append(string)
    n = max([len(bits) for bits in output1+output2])
    output1 = [bits + "1"*(n-len(bits)) for bits in output1]
    output2 = [bits + "1"*(n-len(bits)) for bits in output2]
    return output1, output2


# In[5]:


def expand(texts):
    if len(texts) == 1:
        return texts
    if len(texts) == 2:
        return [text1 + text2 for text1 in texts[0] for text2 in texts[1]]
    if len(texts) > 2:
        return [text1 + text2 for text1 in texts[0] for text2 in expand(texts[1:])]


# In[6]:


#bits1 = ["0*1*"]
#bits2 = ["*1*0"]
p1 = [["",
       "C"]
    ]
p2 = [["C"]
     ]

bits1, bits2 = [], []
for i in range(len(p1)):
    b1, b2 = code(p1[i], p2[i])
    bits1.append(b1)
    bits2.append(b2)

bits1 = expand(bits1)
bits2 = expand(bits2)

bits1 = bits1[0]
bits2 = bits2[0]

n = len(bits1[0]) + 2  #n must >= 4
m = 1  #ans
loop_times = round(math.acos(math.sqrt(m/(2**(n-2))))/(2 * math.sqrt(m/(2**(n-2)))))


# In[7]:


q = QuantumRegister (n)
c = ClassicalRegister (n-2)
qc = QuantumCircuit (q, c)

#initiation
for i in range(n-2):
    qc.h(i)

#Grover's iteration
for i in range(loop_times):
    #Oracles
    for bits in bits1:       #Mark patent 1
        mark(bits, qc, n-2)
    for bits in bits2:       #Mark patent 2
        mark(bits, qc, n-1)
    qc.cz(n-2, n-1)          #Flip the phase
    for bits in bits2[::-1]:       #Disentangle patent 2
        mark(bits, qc, n-1)
    for bits in bits1[::-1]:       #Disentangle patent 1
        mark(bits, qc, n-2)
    
    #Amplitude amplification
    for i in range(n-2):
        qc.h(i)
        qc.x(i)
    qc.cz(0, 1)
    for i in range(n-2):
        qc.x(i)
        qc.h(i)

qc.measure([*range(n-2)], [*range(n-2)])
qc.draw()


# In[17]:


from qiskit.providers.models import backendproperties
ibmq_vigo = provider.get_backend('ibmq_vigo')

thermal_model = NoiseModel.from_backend(ibmq_vigo,
                                      gate_error=False, 
                                      readout_error=False,
                                      thermal_relaxation=True,
                                      temperature=0,
                                      gate_lengths=None,
                                      gate_length_units='ns',
                                      standard_gates=True,
                                      warnings=True)
thermal_simulator = QasmSimulator(noise_model=thermal_model)

print("1x Thermal error:")
job = execute(qc, thermal_simulator,
              basis_gates=thermal_model.basis_gates,
              noise_model=thermal_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, thermal_simulator,
              basis_gates=thermal_model.basis_gates,
              noise_model=thermal_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, thermal_simulator,
              basis_gates=thermal_model.basis_gates,
              noise_model=thermal_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, thermal_simulator,
              basis_gates=thermal_model.basis_gates,
              noise_model=thermal_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, thermal_simulator,
              basis_gates=thermal_model.basis_gates,
              noise_model=thermal_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, thermal_simulator,
              basis_gates=thermal_model.basis_gates,
              noise_model=thermal_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, thermal_simulator,
              basis_gates=thermal_model.basis_gates,
              noise_model=thermal_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, thermal_simulator,
              basis_gates=thermal_model.basis_gates,
              noise_model=thermal_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, thermal_simulator,
              basis_gates=thermal_model.basis_gates,
              noise_model=thermal_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, thermal_simulator,
              basis_gates=thermal_model.basis_gates,
              noise_model=thermal_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)


# In[18]:


t1s = []
t2s = []
t1s.append(ibmq_vigo.properties().t1(0))
t2s.append(ibmq_vigo.properties().t2(0))
t1s.append(ibmq_vigo.properties().t1(1))
t2s.append(ibmq_vigo.properties().t2(1))
t1s.append(ibmq_vigo.properties().t1(2))
t2s.append(ibmq_vigo.properties().t2(2))
t1s.append(ibmq_vigo.properties().t1(3))
t2s.append(ibmq_vigo.properties().t2(3))
t1s.append(ibmq_vigo.properties().t1(4))
t2s.append(ibmq_vigo.properties().t2(4))

print(t1s, t2s)


# In[19]:


thermal2_model = NoiseModel.from_backend(ibmq_vigo,
                                      gate_error=False, 
                                      readout_error=False,
                                      thermal_relaxation=False,
                                      temperature=0,
                                      gate_lengths=None,
                                      gate_length_units='ns',
                                      standard_gates=True,
                                      warnings=True)

# Halve T1 and T2 values (unit = nanosecond)
T1s, T2s = [], []
T1s.append(t1s[0]*0.5e9)
T2s.append(t2s[0]*0.5e9)
T1s.append(t1s[1]*0.5e9)
T2s.append(t2s[1]*0.5e9)
T1s.append(t1s[2]*0.5e9)
T2s.append(t2s[2]*0.5e9)
T1s.append(t1s[3]*0.5e9)
T2s.append(t2s[3]*0.5e9)
T1s.append(t1s[4]*0.5e9)
T2s.append(t2s[4]*0.5e9)

# Truncate random T2s <= T1s
T2s = list(np.array([min(T2s[j], 2*T1s[j]) for j in range(5)]))

# Instruction times (in nanoseconds)
time_u1 = 0   # virtual gate
time_u2 = 35.555555555555554  # (single X90 pulse)
time_u3 = 71.11111111111111 # (two X90 pulses)
time_cx = 519.1111111111111

errors_u1  = [thermal_relaxation_error(T1, T2, time_u1)
              for T1, T2 in zip(T1s, T2s)]
errors_u2  = [thermal_relaxation_error(T1, T2, time_u2)
              for T1, T2 in zip(T1s, T2s)]
errors_u3  = [thermal_relaxation_error(T1, T2, time_u3)
              for T1, T2 in zip(T1s, T2s)]
errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(
             thermal_relaxation_error(t1b, t2b, time_cx))
              for t1a, t2a in zip(T1s, T2s)]
               for t1b, t2b in zip(T1s, T2s)]
thermal2_model.add_quantum_error(errors_u1[0], "u1", [0])
thermal2_model.add_quantum_error(errors_u2[0], "u2", [0])
thermal2_model.add_quantum_error(errors_u3[0], "u3", [0])
thermal2_model.add_quantum_error(errors_cx[0][0], "cx", [0, 0])
thermal2_model.add_quantum_error(errors_cx[0][1], "cx", [0, 1])
thermal2_model.add_quantum_error(errors_cx[0][2], "cx", [0, 2])
thermal2_model.add_quantum_error(errors_cx[0][3], "cx", [0, 3])
thermal2_model.add_quantum_error(errors_u1[1], "u1", [1])
thermal2_model.add_quantum_error(errors_u2[1], "u2", [1])
thermal2_model.add_quantum_error(errors_u3[1], "u3", [1])
thermal2_model.add_quantum_error(errors_cx[1][0], "cx", [1, 0])
thermal2_model.add_quantum_error(errors_cx[1][1], "cx", [1, 1])
thermal2_model.add_quantum_error(errors_cx[1][2], "cx", [1, 2])
thermal2_model.add_quantum_error(errors_cx[1][3], "cx", [1, 3])
thermal2_model.add_quantum_error(errors_u1[2], "u1", [2])
thermal2_model.add_quantum_error(errors_u2[2], "u2", [2])
thermal2_model.add_quantum_error(errors_u3[2], "u3", [2])
thermal2_model.add_quantum_error(errors_cx[2][0], "cx", [2, 0])
thermal2_model.add_quantum_error(errors_cx[2][1], "cx", [2, 1])
thermal2_model.add_quantum_error(errors_cx[2][2], "cx", [2, 2])
thermal2_model.add_quantum_error(errors_cx[2][3], "cx", [2, 3])
thermal2_model.add_quantum_error(errors_u1[3], "u1", [3])
thermal2_model.add_quantum_error(errors_u2[3], "u2", [3])
thermal2_model.add_quantum_error(errors_u3[3], "u3", [3])
thermal2_model.add_quantum_error(errors_cx[3][0], "cx", [3, 0])
thermal2_model.add_quantum_error(errors_cx[3][1], "cx", [3, 1])
thermal2_model.add_quantum_error(errors_cx[3][2], "cx", [3, 2])
thermal2_model.add_quantum_error(errors_cx[3][3], "cx", [3, 3])

print("2x Thermal error:")
thermal2_simulator = QasmSimulator(noise_model=thermal2_model)
job = execute(qc, thermal2_simulator,
              basis_gates=thermal2_model.basis_gates,
              noise_model=thermal2_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, thermal2_simulator,
              basis_gates=thermal2_model.basis_gates,
              noise_model=thermal2_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, thermal2_simulator,
              basis_gates=thermal2_model.basis_gates,
              noise_model=thermal2_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, thermal2_simulator,
              basis_gates=thermal2_model.basis_gates,
              noise_model=thermal2_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, thermal2_simulator,
              basis_gates=thermal2_model.basis_gates,
              noise_model=thermal2_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, thermal2_simulator,
              basis_gates=thermal2_model.basis_gates,
              noise_model=thermal2_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, thermal2_simulator,
              basis_gates=thermal2_model.basis_gates,
              noise_model=thermal2_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, thermal2_simulator,
              basis_gates=thermal2_model.basis_gates,
              noise_model=thermal2_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, thermal2_simulator,
              basis_gates=thermal2_model.basis_gates,
              noise_model=thermal2_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, thermal2_simulator,
              basis_gates=thermal2_model.basis_gates,
              noise_model=thermal2_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)


# In[22]:


thermal2_model = NoiseModel.from_backend(ibmq_vigo,
                                      gate_error=False, 
                                      readout_error=False,
                                      thermal_relaxation=False,
                                      temperature=0,
                                      gate_lengths=None,
                                      gate_length_units='ns',
                                      standard_gates=True,
                                      warnings=True)

# Double T1 and T2 values (unit = nanosecond)
T1s, T2s = [], []
T1s.append(t1s[0]*2e9)
T2s.append(t2s[0]*2e9)
T1s.append(t1s[1]*2e9)
T2s.append(t2s[1]*2e9)
T1s.append(t1s[2]*2e9)
T2s.append(t2s[2]*2e9)
T1s.append(t1s[3]*2e9)
T2s.append(t2s[3]*2e9)
T1s.append(t1s[4]*2e9)
T2s.append(t2s[4]*2e9)

# Truncate random T2s <= T1s
T2s = list(np.array([min(T2s[j], 2*T1s[j]) for j in range(5)]))

# Instruction times (in nanoseconds)
time_u1 = 0   # virtual gate
time_u2 = 35.555555555555554  # (single X90 pulse)
time_u3 = 71.11111111111111 # (two X90 pulses)
time_cx = 519.1111111111111

errors_u1  = [thermal_relaxation_error(T1, T2, time_u1)
              for T1, T2 in zip(T1s, T2s)]
errors_u2  = [thermal_relaxation_error(T1, T2, time_u2)
              for T1, T2 in zip(T1s, T2s)]
errors_u3  = [thermal_relaxation_error(T1, T2, time_u3)
              for T1, T2 in zip(T1s, T2s)]
errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(
             thermal_relaxation_error(t1b, t2b, time_cx))
              for t1a, t2a in zip(T1s, T2s)]
               for t1b, t2b in zip(T1s, T2s)]
thermal2_model.add_quantum_error(errors_u1[0], "u1", [0])
thermal2_model.add_quantum_error(errors_u2[0], "u2", [0])
thermal2_model.add_quantum_error(errors_u3[0], "u3", [0])
thermal2_model.add_quantum_error(errors_cx[0][0], "cx", [0, 0])
thermal2_model.add_quantum_error(errors_cx[0][1], "cx", [0, 1])
thermal2_model.add_quantum_error(errors_cx[0][2], "cx", [0, 2])
thermal2_model.add_quantum_error(errors_cx[0][3], "cx", [0, 3])
thermal2_model.add_quantum_error(errors_u1[1], "u1", [1])
thermal2_model.add_quantum_error(errors_u2[1], "u2", [1])
thermal2_model.add_quantum_error(errors_u3[1], "u3", [1])
thermal2_model.add_quantum_error(errors_cx[1][0], "cx", [1, 0])
thermal2_model.add_quantum_error(errors_cx[1][1], "cx", [1, 1])
thermal2_model.add_quantum_error(errors_cx[1][2], "cx", [1, 2])
thermal2_model.add_quantum_error(errors_cx[1][3], "cx", [1, 3])
thermal2_model.add_quantum_error(errors_u1[2], "u1", [2])
thermal2_model.add_quantum_error(errors_u2[2], "u2", [2])
thermal2_model.add_quantum_error(errors_u3[2], "u3", [2])
thermal2_model.add_quantum_error(errors_cx[2][0], "cx", [2, 0])
thermal2_model.add_quantum_error(errors_cx[2][1], "cx", [2, 1])
thermal2_model.add_quantum_error(errors_cx[2][2], "cx", [2, 2])
thermal2_model.add_quantum_error(errors_cx[2][3], "cx", [2, 3])
thermal2_model.add_quantum_error(errors_u1[3], "u1", [3])
thermal2_model.add_quantum_error(errors_u2[3], "u2", [3])
thermal2_model.add_quantum_error(errors_u3[3], "u3", [3])
thermal2_model.add_quantum_error(errors_cx[3][0], "cx", [3, 0])
thermal2_model.add_quantum_error(errors_cx[3][1], "cx", [3, 1])
thermal2_model.add_quantum_error(errors_cx[3][2], "cx", [3, 2])
thermal2_model.add_quantum_error(errors_cx[3][3], "cx", [3, 3])

print("0.5x Thermal error:")
thermal2_simulator = QasmSimulator(noise_model=thermal2_model)
job = execute(qc, thermal2_simulator,
              basis_gates=thermal2_model.basis_gates,
              noise_model=thermal2_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, thermal2_simulator,
              basis_gates=thermal2_model.basis_gates,
              noise_model=thermal2_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, thermal2_simulator,
              basis_gates=thermal2_model.basis_gates,
              noise_model=thermal2_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, thermal2_simulator,
              basis_gates=thermal2_model.basis_gates,
              noise_model=thermal2_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, thermal2_simulator,
              basis_gates=thermal2_model.basis_gates,
              noise_model=thermal2_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, thermal2_simulator,
              basis_gates=thermal2_model.basis_gates,
              noise_model=thermal2_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, thermal2_simulator,
              basis_gates=thermal2_model.basis_gates,
              noise_model=thermal2_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, thermal2_simulator,
              basis_gates=thermal2_model.basis_gates,
              noise_model=thermal2_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, thermal2_simulator,
              basis_gates=thermal2_model.basis_gates,
              noise_model=thermal2_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, thermal2_simulator,
              basis_gates=thermal2_model.basis_gates,
              noise_model=thermal2_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)


# In[23]:


depolarizing_model = NoiseModel.from_backend(ibmq_vigo,
                                      gate_error=True, 
                                      readout_error=False,
                                      thermal_relaxation=False,
                                      temperature=0,
                                      gate_lengths=None,
                                      gate_length_units='ns',
                                      standard_gates=True,
                                      warnings=True)
depolarizing_simulator = QasmSimulator(noise_model=depolarizing_model)

print("1x Depolarizing error:")
job = execute(qc, depolarizing_simulator,
              basis_gates=depolarizing_model.basis_gates,
              noise_model=depolarizing_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, depolarizing_simulator,
              basis_gates=depolarizing_model.basis_gates,
              noise_model=depolarizing_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, depolarizing_simulator,
              basis_gates=depolarizing_model.basis_gates,
              noise_model=depolarizing_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, depolarizing_simulator,
              basis_gates=depolarizing_model.basis_gates,
              noise_model=depolarizing_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, depolarizing_simulator,
              basis_gates=depolarizing_model.basis_gates,
              noise_model=depolarizing_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, depolarizing_simulator,
              basis_gates=depolarizing_model.basis_gates,
              noise_model=depolarizing_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, depolarizing_simulator,
              basis_gates=depolarizing_model.basis_gates,
              noise_model=depolarizing_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, depolarizing_simulator,
              basis_gates=depolarizing_model.basis_gates,
              noise_model=depolarizing_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, depolarizing_simulator,
              basis_gates=depolarizing_model.basis_gates,
              noise_model=depolarizing_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, depolarizing_simulator,
              basis_gates=depolarizing_model.basis_gates,
              noise_model=depolarizing_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)


# In[24]:


depolarizing_model = NoiseModel.from_backend(ibmq_vigo,
                                      gate_error=False, 
                                      readout_error=False,
                                      thermal_relaxation=False,
                                      temperature=0,
                                      gate_lengths=None,
                                      gate_length_units='ns',
                                      standard_gates=True,
                                      warnings=True)

error = depolarizing_error(2*0.006875, 1) #0.006875=1x
error2 = depolarizing_error(2*0.006875, 2)

depolarizing_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])
depolarizing_model.add_all_qubit_quantum_error(error2, ['cx'])

depolarizing_simulator = QasmSimulator(noise_model=depolarizing_model)
print("2x Depolarizing error:")
job = execute(qc, depolarizing_simulator,
              basis_gates=depolarizing_model.basis_gates,
              noise_model=depolarizing_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, depolarizing_simulator,
              basis_gates=depolarizing_model.basis_gates,
              noise_model=depolarizing_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, depolarizing_simulator,
              basis_gates=depolarizing_model.basis_gates,
              noise_model=depolarizing_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, depolarizing_simulator,
              basis_gates=depolarizing_model.basis_gates,
              noise_model=depolarizing_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, depolarizing_simulator,
              basis_gates=depolarizing_model.basis_gates,
              noise_model=depolarizing_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, depolarizing_simulator,
              basis_gates=depolarizing_model.basis_gates,
              noise_model=depolarizing_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, depolarizing_simulator,
              basis_gates=depolarizing_model.basis_gates,
              noise_model=depolarizing_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, depolarizing_simulator,
              basis_gates=depolarizing_model.basis_gates,
              noise_model=depolarizing_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, depolarizing_simulator,
              basis_gates=depolarizing_model.basis_gates,
              noise_model=depolarizing_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, depolarizing_simulator,
              basis_gates=depolarizing_model.basis_gates,
              noise_model=depolarizing_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)


# In[25]:


depolarizing_model = NoiseModel.from_backend(ibmq_vigo,
                                      gate_error=False, 
                                      readout_error=False,
                                      thermal_relaxation=False,
                                      temperature=0,
                                      gate_lengths=None,
                                      gate_length_units='ns',
                                      standard_gates=True,
                                      warnings=True)

error = depolarizing_error(0.5*0.006875, 1) #0.006875=1x
error2 = depolarizing_error(0.5*0.006875, 2)

depolarizing_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])
depolarizing_model.add_all_qubit_quantum_error(error2, ['cx'])

depolarizing_simulator = QasmSimulator(noise_model=depolarizing_model)
print("0.5x Depolarizing error:")
job = execute(qc, depolarizing_simulator,
              basis_gates=depolarizing_model.basis_gates,
              noise_model=depolarizing_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, depolarizing_simulator,
              basis_gates=depolarizing_model.basis_gates,
              noise_model=depolarizing_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, depolarizing_simulator,
              basis_gates=depolarizing_model.basis_gates,
              noise_model=depolarizing_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, depolarizing_simulator,
              basis_gates=depolarizing_model.basis_gates,
              noise_model=depolarizing_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, depolarizing_simulator,
              basis_gates=depolarizing_model.basis_gates,
              noise_model=depolarizing_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, depolarizing_simulator,
              basis_gates=depolarizing_model.basis_gates,
              noise_model=depolarizing_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, depolarizing_simulator,
              basis_gates=depolarizing_model.basis_gates,
              noise_model=depolarizing_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, depolarizing_simulator,
              basis_gates=depolarizing_model.basis_gates,
              noise_model=depolarizing_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, depolarizing_simulator,
              basis_gates=depolarizing_model.basis_gates,
              noise_model=depolarizing_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, depolarizing_simulator,
              basis_gates=depolarizing_model.basis_gates,
              noise_model=depolarizing_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)


# In[26]:


errors = []
errors.append(ibmq_vigo.properties().readout_error(0))
print(errors[0])
errors.append(ibmq_vigo.properties().readout_error(1))
print(errors[1])
errors.append(ibmq_vigo.properties().readout_error(2))
print(errors[2])
errors.append(ibmq_vigo.properties().readout_error(3))
print(errors[3])
errors.append(ibmq_vigo.properties().readout_error(4))
print(errors[4])


# In[27]:


readout_model = NoiseModel.from_backend(ibmq_vigo,
                                      gate_error=False, 
                                      readout_error=True,
                                      thermal_relaxation=False,
                                      temperature=0,
                                      gate_lengths=None,
                                      gate_length_units='ns',
                                      standard_gates=True,
                                      warnings=True)

readout_simulator = QasmSimulator(noise_model=readout_model)

print("1x Readout error:")
job = execute(qc, readout_simulator,
              basis_gates=readout_model.basis_gates,
              noise_model=readout_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, readout_simulator,
              basis_gates=readout_model.basis_gates,
              noise_model=readout_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, readout_simulator,
              basis_gates=readout_model.basis_gates,
              noise_model=readout_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, readout_simulator,
              basis_gates=readout_model.basis_gates,
              noise_model=readout_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, readout_simulator,
              basis_gates=readout_model.basis_gates,
              noise_model=readout_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, readout_simulator,
              basis_gates=readout_model.basis_gates,
              noise_model=readout_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, readout_simulator,
              basis_gates=readout_model.basis_gates,
              noise_model=readout_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, readout_simulator,
              basis_gates=readout_model.basis_gates,
              noise_model=readout_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, readout_simulator,
              basis_gates=readout_model.basis_gates,
              noise_model=readout_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, readout_simulator,
              basis_gates=readout_model.basis_gates,
              noise_model=readout_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)


# In[28]:


readout_model = NoiseModel.from_backend(ibmq_vigo,
                                      gate_error=False, 
                                      readout_error=False,
                                      thermal_relaxation=False,
                                      temperature=0,
                                      gate_lengths=None,
                                      gate_length_units='ns',
                                      standard_gates=True,
                                      warnings=True)

error = 2*0.0065
probabilities = []
probabilities.append([1-error, error])
probabilities.append([error, 1-error])
readout_error = ReadoutError(probabilities)
readout_model.add_all_qubit_readout_error(readout_error)

readout_simulator = QasmSimulator(noise_model=readout_model)

print("2x Readout error:")
job = execute(qc, readout_simulator,
              basis_gates=readout_model.basis_gates,
              noise_model=readout_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, readout_simulator,
              basis_gates=readout_model.basis_gates,
              noise_model=readout_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, readout_simulator,
              basis_gates=readout_model.basis_gates,
              noise_model=readout_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, readout_simulator,
              basis_gates=readout_model.basis_gates,
              noise_model=readout_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, readout_simulator,
              basis_gates=readout_model.basis_gates,
              noise_model=readout_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, readout_simulator,
              basis_gates=readout_model.basis_gates,
              noise_model=readout_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, readout_simulator,
              basis_gates=readout_model.basis_gates,
              noise_model=readout_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, readout_simulator,
              basis_gates=readout_model.basis_gates,
              noise_model=readout_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, readout_simulator,
              basis_gates=readout_model.basis_gates,
              noise_model=readout_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, readout_simulator,
              basis_gates=readout_model.basis_gates,
              noise_model=readout_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)


# In[29]:


readout_model = NoiseModel.from_backend(ibmq_vigo,
                                      gate_error=False, 
                                      readout_error=False,
                                      thermal_relaxation=False,
                                      temperature=0,
                                      gate_lengths=None,
                                      gate_length_units='ns',
                                      standard_gates=True,
                                      warnings=True)

error = 0.5*0.0065
probabilities = []
probabilities.append([1-error, error])
probabilities.append([error, 1-error])
readout_error = ReadoutError(probabilities)
readout_model.add_all_qubit_readout_error(readout_error)

readout_simulator = QasmSimulator(noise_model=readout_model)

print("0.5x Readout error:")
job = execute(qc, readout_simulator,
              basis_gates=readout_model.basis_gates,
              noise_model=readout_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, readout_simulator,
              basis_gates=readout_model.basis_gates,
              noise_model=readout_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, readout_simulator,
              basis_gates=readout_model.basis_gates,
              noise_model=readout_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, readout_simulator,
              basis_gates=readout_model.basis_gates,
              noise_model=readout_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, readout_simulator,
              basis_gates=readout_model.basis_gates,
              noise_model=readout_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, readout_simulator,
              basis_gates=readout_model.basis_gates,
              noise_model=readout_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, readout_simulator,
              basis_gates=readout_model.basis_gates,
              noise_model=readout_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, readout_simulator,
              basis_gates=readout_model.basis_gates,
              noise_model=readout_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, readout_simulator,
              basis_gates=readout_model.basis_gates,
              noise_model=readout_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)
job = execute(qc, readout_simulator,
              basis_gates=readout_model.basis_gates,
              noise_model=readout_model,
              shots=1024,
              optimization_level=3, 
              initial_layout=[1, 2, 0, 3])
result = job.result()
output = result.get_counts(0)
print(output)


# In[ ]:




