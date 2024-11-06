# https://github.com/SPL-LSU/Codes/blob/a28db11b399e6175134e55b973997b67fa44b0df/Quantum%20Teleportation%20Code%20(Fixed)%20(1).py
#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *

# Loading your IBM Q account(s)
TOKEN ='4bb7bfab446013a8f953525f3692437a0a314f4f2784431d78ee3eaacd042dfbba453ede40f604c8c996197cde23c9fb509d22c8132c5ac0fd3ffe10c5885bf4'
IBMQ.load_account() # Load account from disk
providers = IBMQ.providers()
provider = IBMQ.get_provider(hub ='ibm-q')
print(provider)
print(provider.backends())
backend = provider.get_backend('ibmq_london')







# In[9]:


from qiskit import *
from qiskit.providers.aer import noise, QasmSimulator, StatevectorSimulator, UnitarySimulator
from qiskit.providers.aer.noise import NoiseModel

#Using simulator from Qiskit Aer
backend_sim = Aer.get_backend('qasm_simulator')



#Using real machine as a backend

#provider = IBMQ.get_provider(hub='ibm-q')
#backend_sim = provider.get_backend('ibmq_london')



#Using real machines as noise model

#provider = IBMQ.get_provider(hub = 'ibm-q') 
#device = provider.get_backend('ibmq_16_melbourne') 
#device = provider.get_backend('ibmqx2') 
#device = provider.get_backend('ibmq_essex')

#properties = device.properties() #will be used to help generate a noise model to run on the simulator
#coupling_map = device.configuration().coupling_map 

number_of_counts = {} #Dictionary for the total number of shots of each state obtained 
number_of_noisy_counts = {} #Dictionary for the total number of shots of each state with noise obtained

#This function returns an image of the circuit that is being run
# Returns-Image of circuit
def showTeleportationCircuit(qc):
    circuit = qc.draw()
    return circuit
    
#This function teleports a state from qubit one to qubit three
#Returns-nothing
def teleportState(qc, q):
    qc.h(q[1])
    qc.cx(q[1],q[2])
    qc.cx(q[0],q[1])
    qc.h(q[0])
    qc.cx(q[1],q[2])
    qc.cz(q[0],q[2])

#This function creates the qubit state one from default state 0
#Returns-nothing
def createOneState(qc, q):
    qc.x(q[0])

#This function creates the qubit plus state from default state 0
#Returns-nothing
def createPlusState(qc, q):
    qc.h(q[0])

#This function creates the qubit minus state from default state 0
#Returns-nothing
def createMinusState(qc, q):
    qc.h(q[0])
    qc.z(q[0])

#This function creates the qubit state plus i from default state 0
def createPlusIState(qc, q):
    qc.h(q[0])
    qc.s(q[0])

#This function creates the qubit state minus i from default state 0
#Returns-nothing
def createMinusIState(qc, q):
    qc.h(q[0])
    qc.z(q[0])
    qc.s(q[0])
    
def createRandomState(qc, q, p):
    qc.u3(p, 0 , 0, q[0])

#This function creates a noise model to apply to a circuit 
# Returns- the amount of counts for each state with noise
def applyNoise(qc):
    gate_lengths = [
    ('u1', None, 0), ('u2', None, 100), ('u3', None, 200),
    ('cx', [1, 0], 678), ('cx', [1, 2], 547), ('cx', [2, 3], 721),
    ('cx', [4, 3], 733), ('cx', [4, 10], 721), ('cx', [5, 4], 800),
    ('cx', [5, 6], 800), ('cx', [5, 9], 895), ('cx', [6, 8], 895),
    ('cx', [7, 8], 640), ('cx', [9, 8], 895), ('cx', [9, 10], 800),
    ('cx', [11, 10], 721), ('cx', [11, 3], 634), ('cx', [12, 2], 773),
    ('cx', [13, 1], 2286), ('cx', [13, 12], 1504), ('cx', [], 800)
   ]
    noise_model = NoiseModel.from_backend(properties, gate_lengths=gate_lengths)
    #noise_model = noise.device.basic_device_noise_model(properties, gate_lengths=gate_lengths)
    #print(noise_model)
    
    basis_gates = noise_model.basis_gates
    #simulator = Aer.get_backend('qasm_simulator')
    
    
    result_noise = execute(qc, backend_sim, shots = 10000,
                          noise_model = noise_model,
                          coupling_map = coupling_map,
                          basis_gates = basis_gates).result()
    counts_noise = result_noise.get_counts(qc)
    return counts_noise

#This function teleports the initial qubit state 0 from qubit 1 to qubit 3 both with noise and without noise
#Returns-nothing
def teleportZeroState():
    q0 = QuantumRegister(3)
    c0 = ClassicalRegister(1)
    qc0 = QuantumCircuit(q0,c0)
    teleportState(qc0, q0)
    qc0.measure(q0[2],c0)
    job = execute(qc0, backend_sim, shots=10000) #make sure to remove shots when running on actual machine
    counts = job.result().get_counts()
    
    print("ZeroState: ", counts)
    number_of_counts['ZeroState'] = counts
    
    
    #Applying noise
    #noise = applyNoise(qc0)
    #print("NoisyZeroState: ", noise)
    #number_of_noisy_counts['ZeroState']= noise
    
#This function teleports the initial qubit state 1 from qubit 1 to qubit 3 both with noise and without noise
#Returns-nothing    
def teleportOneState():
    q1 = QuantumRegister(3)
    c1 = ClassicalRegister(1)
    qc1 = QuantumCircuit(q1,c1)
    createOneState(qc1, q1)
    teleportState(qc1, q1)
    qc1.measure(q1[2],c1)
    job = execute(qc1, backend_sim, shots = 10000)
    counts = job.result().get_counts(qc1)
 
    print("OneState: ", counts)
    number_of_counts['OneState'] = counts
    
    #Applying noise
    #noise = applyNoise(qc1)
    #print("NoisyOneState: ", noise)
    #number_of_noisy_counts['OneState']= noise
    

#This function teleports the initial qubit state |+> from qubit 1 to qubit 3 both with noise and without noise
#Returns-nothing      
def teleportPlusState():
    qP = QuantumRegister(3)
    cP = ClassicalRegister(1)
    qcP = QuantumCircuit(qP,cP)
    createPlusState(qcP, qP)
    teleportState(qcP, qP)
    qcP.h(qP[2])
    qcP.measure(qP[2],cP)
    job = execute(qcP, backend_sim, shots = 10000)
    counts = job.result().get_counts(qcP)
    
    print("PlusState: ", counts)
    number_of_counts['PlusState']= counts
        
    #Applying noise
    #noise = applyNoise(qcP)
    #print("NoisyPlusState: ", noise)
    #number_of_noisy_counts['PlusState'] = noise

#This function teleports the initial qubit state |-> from qubit 1 to qubit 3 both with noise and without noise
#Returns-nothing   
def teleportMinusState():
    qM = QuantumRegister(3)
    cM = ClassicalRegister(1)
    qcM = QuantumCircuit(qM,cM)
    createMinusState(qcM, qM)
    teleportState(qcM, qM)
    qcM.h(qM[2])
    qcM.measure(qM[2],cM)
    job = execute(qcM, backend_sim, shots = 10000)
    counts = job.result().get_counts(qcM)
    
    print("MinusState: ", counts)
    number_of_counts['MinusState'] = counts
    
    #Applying noise
    #noise = applyNoise(qcM)
    #print("NoisyMinusState: ", noise)
    #number_of_noisy_counts['MinusState'] = noise

#This function teleports the initial qubit state |+i> from qubit 1 to qubit 3 both with noise and without noise
#Returns-nothing
def teleportPlusIState():
    qPI = QuantumRegister(3)
    cPI = ClassicalRegister(1)
    qcPI = QuantumCircuit(qPI,cPI)
    createPlusIState(qcPI, qPI)
    teleportState(qcPI, qPI)
    qcPI.sdg(qPI[2])
    qcPI.h(qPI[2])
    qcPI.measure(qPI[2],cPI)
    job = execute(qcPI, backend_sim, shots = 10000)
    counts = job.result().get_counts(qcPI)

    print("PlusIState: ", counts)
    number_of_counts['PlusIState'] = counts
    
    #Applying noise
    #noise = applyNoise(qcPI)
    #print("NoisyPlusIState: ", noise)
    #number_of_noisy_counts['PlusIState'] = noise

#This function teleports the initial qubit state |-i> from qubit 1 to qubit 3 both with noise and without noise
#Returns-nothing
def teleportMinusIState():
    qMI = QuantumRegister(3)
    cMI = ClassicalRegister(1)
    qcMI = QuantumCircuit(qMI,cMI)
    createMinusIState(qcMI, qMI)
    teleportState(qcMI, qMI)
    qcMI.sdg(qMI[2])
    qcMI.h(qMI[2])
    qcMI.measure(qMI[2],cMI)
    job = execute(qcMI, backend_sim, shots = 10000)
    counts = job.result().get_counts(qcMI)
    
    print("MinusIState: ", counts)
    number_of_counts['MinusIState'] = counts
    
    #Applying noise
    #noise = applyNoise(qcMI)
    #print("NoisyMinusIState: ", noise)
    #number_of_noisy_counts['MinusIState'] = noise
    
    
#This function calculates the fidelity of teleporting an arbitrary state |psi>=sqrt(p)|0>+sqrt(1-p)|1> using:
#p*(fraction of zero outcomes)+(1-p)*(fraction of one outcomes)
#Returns-fidelity of teleporting an arbitrary state
def calculateFidelity(p):
    Values = number_of_noisy_counts['RandomState:|psi>=sqrt(', p, ')|0>+sqrt(1- ', p, ')|1>']
    fidelity = (p* ((Values['0'])/1024)) + ((1-p) * ((Values['1'])/1024))
    return fidelity
    

#This function calculates the average fidelity of teleporting all the pole states: 0, 1, +, -, +i, -i
#Returns-average fidelity
def calculatingFidelity():
    ZeroValues = number_of_counts['ZeroState']
    ZeroProb = (ZeroValues['0'])/10000
    
    OneValues = number_of_counts['OneState']
    OneProb = (OneValues['1'])/10000
    
    PlusValues = number_of_counts['PlusState']
    PlusProbZero = (PlusValues['0'])/10000
    
    MinusValues = number_of_counts['MinusState']
    MinusProbOne = (MinusValues['1'])/10000
    
    PlusIValues = number_of_counts['PlusIState']
    PlusIProbZero = (PlusIValues['0'])/10000
    
    MinusIValues = number_of_counts['MinusIState']
    MinusIProbOne = (MinusIValues['1'])/10000
    
    Fidelity = (ZeroProb + OneProb + PlusProbZero + MinusProbOne + PlusIProbZero + MinusIProbOne)/6
    return Fidelity
    
    
    
    
    
    

def main():      
   print("Trial: ",  (0+1))
   teleportZeroState()
   teleportOneState()
   teleportPlusState()
   teleportMinusState()
   teleportPlusIState()
   teleportMinusIState()
   fid = calculatingFidelity()
   print(fid)
   print("-----------------------------------------")
   print("Trial: ",  (1+1))
   teleportZeroState()
   teleportOneState()
   teleportPlusState()
   teleportMinusState()
   teleportPlusIState()
   teleportMinusIState()
   fid = calculatingFidelity()
   print(fid)
   print("-----------------------------------------")
   print("Trial: ",  (2+1))
   teleportZeroState()
   teleportOneState()
   teleportPlusState()
   teleportMinusState()
   teleportPlusIState()
   teleportMinusIState()
   fid = calculatingFidelity()
   print(fid)
   print("-----------------------------------------")
   print("Trial: ",  (3+1))
   teleportZeroState()
   teleportOneState()
   teleportPlusState()
   teleportMinusState()
   teleportPlusIState()
   teleportMinusIState()
   fid = calculatingFidelity()
   print(fid)
   print("-----------------------------------------")
   print("Trial: ",  (4+1))
   teleportZeroState()
   teleportOneState()
   teleportPlusState()
   teleportMinusState()
   teleportPlusIState()
   teleportMinusIState()
   fid = calculatingFidelity()
   print(fid)
   print("-----------------------------------------")
   print("Trial: ",  (5+1))
   teleportZeroState()
   teleportOneState()
   teleportPlusState()
   teleportMinusState()
   teleportPlusIState()
   teleportMinusIState()
   fid = calculatingFidelity()
   print(fid)
   print("-----------------------------------------")
   print("Trial: ",  (6+1))
   teleportZeroState()
   teleportOneState()
   teleportPlusState()
   teleportMinusState()
   teleportPlusIState()
   teleportMinusIState()
   fid = calculatingFidelity()
   print(fid)
   print("-----------------------------------------")
   print("Trial: ",  (7+1))
   teleportZeroState()
   teleportOneState()
   teleportPlusState()
   teleportMinusState()
   teleportPlusIState()
   teleportMinusIState()
   fid = calculatingFidelity()
   print(fid)
   print("-----------------------------------------")
   print("Trial: ",  (8+1))
   teleportZeroState()
   teleportOneState()
   teleportPlusState()
   teleportMinusState()
   teleportPlusIState()
   teleportMinusIState()
   fid = calculatingFidelity()
   print(fid)
   print("-----------------------------------------")
   print("Trial: ",  (9+1))
   teleportZeroState()
   teleportOneState()
   teleportPlusState()
   teleportMinusState()
   teleportPlusIState()
   teleportMinusIState()
   fid = calculatingFidelity()
   print(fid)
   print("-----------------------------------------")

main()


# In[ ]:




