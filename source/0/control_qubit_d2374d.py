# https://github.com/gwjacobson/QuantumErrorCorrection/blob/6d8cb6c3ed10399575c562ce76b8131c6f799858/control_qubit.py
from qiskit import *
from error import *
from qiskit.providers.aer import extensions
from qiskit.quantum_info import state_fidelity, DensityMatrix, Statevector
import matplotlib.pyplot as plt 
import random
from qiskit.visualization import plot_histogram

##Single qubit being exposed to random errors from error model.
## 10 time steps are ran, each with 10% prob of error. The fidelity of the
## state is checked after each time step.

#define qubits to test, the probability of finding an error, and number of shots
def control_qubits(qubits, shots): 
    
    fid_array = []
    
    sim = Aer.get_backend('statevector_simulator') #simulator

    #inital quantum state
    qr = QuantumRegister(qubits, 'q')
    cr = ClassicalRegister(qubits, 'c')
    qc = QuantumCircuit(qr, cr)

    #array of state fidelities from each run
    fidelity = 0.0
    avg_fid = 0.0

    #create initial state to compare fidelity
    qobj1 = assemble(qc)
    state1 = sim.run(qobj1).result().get_statevector()

    for i in range(0, shots):
        #qobj1 = assemble(qc)
        #state1 = sim.run(qobj1).result().get_statevector()
                

        prob = random.choices([0,1], weights=[(10-0)/10, 0/10], k=1) #variable probability of error
        bit = random.randint(0, qubits-1) #qubit to apply error to

        if prob[0] == 1:
            phase_flip(qc, bit)
            qc.barrier(qr)
        else:
            qc.barrier(qr)
        

        #run the circuit
        qobj2 = assemble(qc)
        results = sim.run(qobj2).result()
        state2 = results.get_statevector() #state after time step to compare fidelity
        fid = state_fidelity(state1,state2)
        fidelity += fid
        avg_fid = fidelity/shots
    
    fid_array.append(avg_fid)
    
    sim = Aer.get_backend('statevector_simulator') #simulator

    #inital quantum state
    qr = QuantumRegister(qubits, 'q')
    cr = ClassicalRegister(qubits, 'c')
    qc = QuantumCircuit(qr, cr)

    #array of state fidelities from each run
    fidelity = 0.0
    avg_fid = 0.0

    #create initial state to compare fidelity
    qobj1 = assemble(qc)
    state1 = sim.run(qobj1).result().get_statevector()

    for i in range(0, shots):
        #qobj1 = assemble(qc)
        #state1 = sim.run(qobj1).result().get_statevector()
                

        prob = random.choices([0,1], weights=[(10-1)/10, 1/10], k=1) #variable probability of error
        bit = random.randint(0, qubits-1) #qubit to apply error to

        if prob[0] == 1:
            phase_flip(qc, bit)
            qc.barrier(qr)
        else:
            qc.barrier(qr)
        

        #run the circuit
        qobj2 = assemble(qc)
        results = sim.run(qobj2).result()
        state2 = results.get_statevector() #state after time step to compare fidelity
        fid = state_fidelity(state1,state2)
        fidelity += fid
        avg_fid = fidelity/shots
    
    fid_array.append(avg_fid)
    
    sim = Aer.get_backend('statevector_simulator') #simulator

    #inital quantum state
    qr = QuantumRegister(qubits, 'q')
    cr = ClassicalRegister(qubits, 'c')
    qc = QuantumCircuit(qr, cr)

    #array of state fidelities from each run
    fidelity = 0.0
    avg_fid = 0.0

    #create initial state to compare fidelity
    qobj1 = assemble(qc)
    state1 = sim.run(qobj1).result().get_statevector()

    for i in range(0, shots):
        #qobj1 = assemble(qc)
        #state1 = sim.run(qobj1).result().get_statevector()
                

        prob = random.choices([0,1], weights=[(10-2)/10, 2/10], k=1) #variable probability of error
        bit = random.randint(0, qubits-1) #qubit to apply error to

        if prob[0] == 1:
            phase_flip(qc, bit)
            qc.barrier(qr)
        else:
            qc.barrier(qr)
        

        #run the circuit
        qobj2 = assemble(qc)
        results = sim.run(qobj2).result()
        state2 = results.get_statevector() #state after time step to compare fidelity
        fid = state_fidelity(state1,state2)
        fidelity += fid
        avg_fid = fidelity/shots
    
    fid_array.append(avg_fid)
    
    sim = Aer.get_backend('statevector_simulator') #simulator

    #inital quantum state
    qr = QuantumRegister(qubits, 'q')
    cr = ClassicalRegister(qubits, 'c')
    qc = QuantumCircuit(qr, cr)

    #array of state fidelities from each run
    fidelity = 0.0
    avg_fid = 0.0

    #create initial state to compare fidelity
    qobj1 = assemble(qc)
    state1 = sim.run(qobj1).result().get_statevector()

    for i in range(0, shots):
        #qobj1 = assemble(qc)
        #state1 = sim.run(qobj1).result().get_statevector()
                

        prob = random.choices([0,1], weights=[(10-3)/10, 3/10], k=1) #variable probability of error
        bit = random.randint(0, qubits-1) #qubit to apply error to

        if prob[0] == 1:
            phase_flip(qc, bit)
            qc.barrier(qr)
        else:
            qc.barrier(qr)
        

        #run the circuit
        qobj2 = assemble(qc)
        results = sim.run(qobj2).result()
        state2 = results.get_statevector() #state after time step to compare fidelity
        fid = state_fidelity(state1,state2)
        fidelity += fid
        avg_fid = fidelity/shots
    
    fid_array.append(avg_fid)
    
    sim = Aer.get_backend('statevector_simulator') #simulator

    #inital quantum state
    qr = QuantumRegister(qubits, 'q')
    cr = ClassicalRegister(qubits, 'c')
    qc = QuantumCircuit(qr, cr)

    #array of state fidelities from each run
    fidelity = 0.0
    avg_fid = 0.0

    #create initial state to compare fidelity
    qobj1 = assemble(qc)
    state1 = sim.run(qobj1).result().get_statevector()

    for i in range(0, shots):
        #qobj1 = assemble(qc)
        #state1 = sim.run(qobj1).result().get_statevector()
                

        prob = random.choices([0,1], weights=[(10-4)/10, 4/10], k=1) #variable probability of error
        bit = random.randint(0, qubits-1) #qubit to apply error to

        if prob[0] == 1:
            phase_flip(qc, bit)
            qc.barrier(qr)
        else:
            qc.barrier(qr)
        

        #run the circuit
        qobj2 = assemble(qc)
        results = sim.run(qobj2).result()
        state2 = results.get_statevector() #state after time step to compare fidelity
        fid = state_fidelity(state1,state2)
        fidelity += fid
        avg_fid = fidelity/shots
    
    fid_array.append(avg_fid)
    
    sim = Aer.get_backend('statevector_simulator') #simulator

    #inital quantum state
    qr = QuantumRegister(qubits, 'q')
    cr = ClassicalRegister(qubits, 'c')
    qc = QuantumCircuit(qr, cr)

    #array of state fidelities from each run
    fidelity = 0.0
    avg_fid = 0.0

    #create initial state to compare fidelity
    qobj1 = assemble(qc)
    state1 = sim.run(qobj1).result().get_statevector()

    for i in range(0, shots):
        #qobj1 = assemble(qc)
        #state1 = sim.run(qobj1).result().get_statevector()
                

        prob = random.choices([0,1], weights=[(10-5)/10, 5/10], k=1) #variable probability of error
        bit = random.randint(0, qubits-1) #qubit to apply error to

        if prob[0] == 1:
            phase_flip(qc, bit)
            qc.barrier(qr)
        else:
            qc.barrier(qr)
        

        #run the circuit
        qobj2 = assemble(qc)
        results = sim.run(qobj2).result()
        state2 = results.get_statevector() #state after time step to compare fidelity
        fid = state_fidelity(state1,state2)
        fidelity += fid
        avg_fid = fidelity/shots
    
    fid_array.append(avg_fid)
    
    sim = Aer.get_backend('statevector_simulator') #simulator

    #inital quantum state
    qr = QuantumRegister(qubits, 'q')
    cr = ClassicalRegister(qubits, 'c')
    qc = QuantumCircuit(qr, cr)

    #array of state fidelities from each run
    fidelity = 0.0
    avg_fid = 0.0

    #create initial state to compare fidelity
    qobj1 = assemble(qc)
    state1 = sim.run(qobj1).result().get_statevector()

    for i in range(0, shots):
        #qobj1 = assemble(qc)
        #state1 = sim.run(qobj1).result().get_statevector()
                

        prob = random.choices([0,1], weights=[(10-6)/10, 6/10], k=1) #variable probability of error
        bit = random.randint(0, qubits-1) #qubit to apply error to

        if prob[0] == 1:
            phase_flip(qc, bit)
            qc.barrier(qr)
        else:
            qc.barrier(qr)
        

        #run the circuit
        qobj2 = assemble(qc)
        results = sim.run(qobj2).result()
        state2 = results.get_statevector() #state after time step to compare fidelity
        fid = state_fidelity(state1,state2)
        fidelity += fid
        avg_fid = fidelity/shots
    
    fid_array.append(avg_fid)
    
    sim = Aer.get_backend('statevector_simulator') #simulator

    #inital quantum state
    qr = QuantumRegister(qubits, 'q')
    cr = ClassicalRegister(qubits, 'c')
    qc = QuantumCircuit(qr, cr)

    #array of state fidelities from each run
    fidelity = 0.0
    avg_fid = 0.0

    #create initial state to compare fidelity
    qobj1 = assemble(qc)
    state1 = sim.run(qobj1).result().get_statevector()

    for i in range(0, shots):
        #qobj1 = assemble(qc)
        #state1 = sim.run(qobj1).result().get_statevector()
                

        prob = random.choices([0,1], weights=[(10-7)/10, 7/10], k=1) #variable probability of error
        bit = random.randint(0, qubits-1) #qubit to apply error to

        if prob[0] == 1:
            phase_flip(qc, bit)
            qc.barrier(qr)
        else:
            qc.barrier(qr)
        

        #run the circuit
        qobj2 = assemble(qc)
        results = sim.run(qobj2).result()
        state2 = results.get_statevector() #state after time step to compare fidelity
        fid = state_fidelity(state1,state2)
        fidelity += fid
        avg_fid = fidelity/shots
    
    fid_array.append(avg_fid)
    
    sim = Aer.get_backend('statevector_simulator') #simulator

    #inital quantum state
    qr = QuantumRegister(qubits, 'q')
    cr = ClassicalRegister(qubits, 'c')
    qc = QuantumCircuit(qr, cr)

    #array of state fidelities from each run
    fidelity = 0.0
    avg_fid = 0.0

    #create initial state to compare fidelity
    qobj1 = assemble(qc)
    state1 = sim.run(qobj1).result().get_statevector()

    for i in range(0, shots):
        #qobj1 = assemble(qc)
        #state1 = sim.run(qobj1).result().get_statevector()
                

        prob = random.choices([0,1], weights=[(10-8)/10, 8/10], k=1) #variable probability of error
        bit = random.randint(0, qubits-1) #qubit to apply error to

        if prob[0] == 1:
            phase_flip(qc, bit)
            qc.barrier(qr)
        else:
            qc.barrier(qr)
        

        #run the circuit
        qobj2 = assemble(qc)
        results = sim.run(qobj2).result()
        state2 = results.get_statevector() #state after time step to compare fidelity
        fid = state_fidelity(state1,state2)
        fidelity += fid
        avg_fid = fidelity/shots
    
    fid_array.append(avg_fid)
    
    sim = Aer.get_backend('statevector_simulator') #simulator

    #inital quantum state
    qr = QuantumRegister(qubits, 'q')
    cr = ClassicalRegister(qubits, 'c')
    qc = QuantumCircuit(qr, cr)

    #array of state fidelities from each run
    fidelity = 0.0
    avg_fid = 0.0

    #create initial state to compare fidelity
    qobj1 = assemble(qc)
    state1 = sim.run(qobj1).result().get_statevector()

    for i in range(0, shots):
        #qobj1 = assemble(qc)
        #state1 = sim.run(qobj1).result().get_statevector()
                

        prob = random.choices([0,1], weights=[(10-9)/10, 9/10], k=1) #variable probability of error
        bit = random.randint(0, qubits-1) #qubit to apply error to

        if prob[0] == 1:
            phase_flip(qc, bit)
            qc.barrier(qr)
        else:
            qc.barrier(qr)
        

        #run the circuit
        qobj2 = assemble(qc)
        results = sim.run(qobj2).result()
        state2 = results.get_statevector() #state after time step to compare fidelity
        fid = state_fidelity(state1,state2)
        fidelity += fid
        avg_fid = fidelity/shots
    
    fid_array.append(avg_fid)

    return fid_array