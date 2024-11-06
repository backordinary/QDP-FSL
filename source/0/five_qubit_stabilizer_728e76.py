# https://github.com/gwjacobson/QuantumErrorCorrection/blob/6d8cb6c3ed10399575c562ce76b8131c6f799858/five_qubit_stabilizer.py
from qiskit import *
from qiskit.quantum_info import state_fidelity
import random
from error import *
from matplotlib import pyplot as plt

##

def five_qubit_stabilizer(shots):
    five_array = []
    #simulator
    sim = Aer.get_backend('statevector_simulator')

    #inital quantum state
    qr1 = QuantumRegister(4, 'ancilla')
    qr2 = QuantumRegister(5, 'q')
    cr = ClassicalRegister(4, 'c')
    five_qc = QuantumCircuit(qr2, qr1, cr)

    five_fidelity = 0.0
    avg_fid = 0.0

    #qobj1 = assemble(five_qc)
    #state1 = sim.run(qobj1).result().get_statevector()

    for i in range(0, shots):
        qobj1 = assemble(five_qc)
        state1 = sim.run(qobj1).result().get_statevector()

        #encoding of five qubits
        five_qc.z(4)
        five_qc.h(4)
        five_qc.z(4)
        five_qc.cx(4, 3)
        five_qc.h(3)
        five_qc.h(4)
        five_qc.cx(4, 2)
        five_qc.cx(3, 2)
        five_qc.h(2)
        five_qc.cx(4, 1)
        five_qc.cx(2, 1)
        five_qc.h(4)
        five_qc.h(1)
        five_qc.cx(4, 0)
        five_qc.cx(3, 0)
        five_qc.h(4)
        five_qc.cx(2, 0)
        five_qc.h(3)
        #five_qc.barrier([0,1,2,3,4,5,6,7,8])


        #Error on encoded qubits
        bit = random.randint(0,4) #bit to apply error to
        prob = random.choices([0,1], weights=[(10-0)/10, 0/10]) #probability of an error on encoded qubits
        if prob[0] == 1:
            phase_flip(five_qc, bit)
            five_qc.barrier([qr2,qr1])
        else:
            five_qc.barrier([qr2,qr1])


        #this is the five qubit stabilizer (4-8) with ancilla measurements (0-3) target on codewords
        five_qc.h(5)
        five_qc.h(6)
        five_qc.h(7)
        five_qc.h(8)
        five_qc.cz(5, 1)
        five_qc.cx(5, 2)
        five_qc.cx(5, 3)
        five_qc.cz(5, 4)
        five_qc.cz(6, 0)
        five_qc.cz(6, 2)
        five_qc.cx(6, 3)
        five_qc.cx(6, 4)
        five_qc.cx(7, 0)
        five_qc.cz(7, 1)
        five_qc.cz(7, 3)
        five_qc.cx(7, 4)
        five_qc.cx(8, 0)
        five_qc.cx(8, 1)
        five_qc.cz(8, 2)
        five_qc.cz(8, 4)
        five_qc.h(5)
        five_qc.h(6)
        five_qc.h(7)
        five_qc.h(8)
        five_qc.measure(5,0)
        five_qc.measure(6,1)
        five_qc.measure(7,2)
        five_qc.measure(8,3)
        
        #check ancilla measurements
        qobj = assemble(five_qc)
        results = sim.run(qobj).result()

        #do error correction
        five_qc.x(1).c_if(cr, 1)
        five_qc.z(3).c_if(cr, 2)
        five_qc.x(0).c_if(cr, 3)
        five_qc.z(0).c_if(cr, 4)
        five_qc.z(2).c_if(cr, 5)
        five_qc.x(4).c_if(cr, 6)
        five_qc.y(0).c_if(cr, 7)
        five_qc.x(2).c_if(cr, 8)
        five_qc.z(4).c_if(cr, 9)
        five_qc.z(1).c_if(cr, 10)
        five_qc.y(1).c_if(cr, 11)
        five_qc.x(3).c_if(cr, 12)
        five_qc.y(2).c_if(cr, 13)
        five_qc.y(3).c_if(cr, 14)
        five_qc.y(4).c_if(cr, 15)
        

        #decode the five_qc
        #five_qc.barrier([0,1,2,3,4,5,6,7,8])
        five_qc.h(4)
        five_qc.cx(2, 0)
        five_qc.h(3)
        five_qc.cx(3, 0)
        five_qc.cx(4, 0)
        five_qc.h(1)
        five_qc.cx(2, 1)
        five_qc.h(4)
        five_qc.cx(4, 1)
        five_qc.h(2)
        five_qc.cx(3, 2)
        five_qc.cx(4, 2)
        five_qc.h(4)
        five_qc.h(3)
        five_qc.cx(4, 3)
        five_qc.z(4)
        five_qc.h(4)
        five_qc.z(4)

        #run five_qc
        qobj2 = assemble(five_qc)
        state2 = sim.run(qobj2).result().get_statevector() #state after time step to compare fidelity
        fid = state_fidelity(state1,state2)
        five_fidelity += fid
        avg_fid = five_fidelity/shots
    
    five_array.append(avg_fid)
    #simulator
    sim = Aer.get_backend('statevector_simulator')

    #inital quantum state
    qr1 = QuantumRegister(4, 'ancilla')
    qr2 = QuantumRegister(5, 'q')
    cr = ClassicalRegister(4, 'c')
    five_qc = QuantumCircuit(qr2, qr1, cr)

    five_fidelity = 0.0
    avg_fid = 0.0

    #qobj1 = assemble(five_qc)
    #state1 = sim.run(qobj1).result().get_statevector()

    for i in range(0, shots):
        qobj1 = assemble(five_qc)
        state1 = sim.run(qobj1).result().get_statevector()

        #encoding of five qubits
        five_qc.z(4)
        five_qc.h(4)
        five_qc.z(4)
        five_qc.cx(4, 3)
        five_qc.h(3)
        five_qc.h(4)
        five_qc.cx(4, 2)
        five_qc.cx(3, 2)
        five_qc.h(2)
        five_qc.cx(4, 1)
        five_qc.cx(2, 1)
        five_qc.h(4)
        five_qc.h(1)
        five_qc.cx(4, 0)
        five_qc.cx(3, 0)
        five_qc.h(4)
        five_qc.cx(2, 0)
        five_qc.h(3)
        #five_qc.barrier([0,1,2,3,4,5,6,7,8])


        #Error on encoded qubits
        bit = random.randint(0,4) #bit to apply error to
        prob = random.choices([0,1], weights=[(10-1)/10, 1/10]) #probability of an error on encoded qubits
        if prob[0] == 1:
            phase_flip(five_qc, bit)
            five_qc.barrier([qr2,qr1])
        else:
            five_qc.barrier([qr2,qr1])


        #this is the five qubit stabilizer (4-8) with ancilla measurements (0-3) target on codewords
        five_qc.h(5)
        five_qc.h(6)
        five_qc.h(7)
        five_qc.h(8)
        five_qc.cz(5, 1)
        five_qc.cx(5, 2)
        five_qc.cx(5, 3)
        five_qc.cz(5, 4)
        five_qc.cz(6, 0)
        five_qc.cz(6, 2)
        five_qc.cx(6, 3)
        five_qc.cx(6, 4)
        five_qc.cx(7, 0)
        five_qc.cz(7, 1)
        five_qc.cz(7, 3)
        five_qc.cx(7, 4)
        five_qc.cx(8, 0)
        five_qc.cx(8, 1)
        five_qc.cz(8, 2)
        five_qc.cz(8, 4)
        five_qc.h(5)
        five_qc.h(6)
        five_qc.h(7)
        five_qc.h(8)
        five_qc.measure(5,0)
        five_qc.measure(6,1)
        five_qc.measure(7,2)
        five_qc.measure(8,3)
        
        #check ancilla measurements
        qobj = assemble(five_qc)
        results = sim.run(qobj).result()

        #do error correction
        five_qc.x(1).c_if(cr, 1)
        five_qc.z(3).c_if(cr, 2)
        five_qc.x(0).c_if(cr, 3)
        five_qc.z(0).c_if(cr, 4)
        five_qc.z(2).c_if(cr, 5)
        five_qc.x(4).c_if(cr, 6)
        five_qc.y(0).c_if(cr, 7)
        five_qc.x(2).c_if(cr, 8)
        five_qc.z(4).c_if(cr, 9)
        five_qc.z(1).c_if(cr, 10)
        five_qc.y(1).c_if(cr, 11)
        five_qc.x(3).c_if(cr, 12)
        five_qc.y(2).c_if(cr, 13)
        five_qc.y(3).c_if(cr, 14)
        five_qc.y(4).c_if(cr, 15)
        

        #decode the five_qc
        #five_qc.barrier([0,1,2,3,4,5,6,7,8])
        five_qc.h(4)
        five_qc.cx(2, 0)
        five_qc.h(3)
        five_qc.cx(3, 0)
        five_qc.cx(4, 0)
        five_qc.h(1)
        five_qc.cx(2, 1)
        five_qc.h(4)
        five_qc.cx(4, 1)
        five_qc.h(2)
        five_qc.cx(3, 2)
        five_qc.cx(4, 2)
        five_qc.h(4)
        five_qc.h(3)
        five_qc.cx(4, 3)
        five_qc.z(4)
        five_qc.h(4)
        five_qc.z(4)

        #run five_qc
        qobj2 = assemble(five_qc)
        state2 = sim.run(qobj2).result().get_statevector() #state after time step to compare fidelity
        fid = state_fidelity(state1,state2)
        five_fidelity += fid
        avg_fid = five_fidelity/shots
    
    five_array.append(avg_fid)
    #simulator
    sim = Aer.get_backend('statevector_simulator')

    #inital quantum state
    qr1 = QuantumRegister(4, 'ancilla')
    qr2 = QuantumRegister(5, 'q')
    cr = ClassicalRegister(4, 'c')
    five_qc = QuantumCircuit(qr2, qr1, cr)

    five_fidelity = 0.0
    avg_fid = 0.0

    #qobj1 = assemble(five_qc)
    #state1 = sim.run(qobj1).result().get_statevector()

    for i in range(0, shots):
        qobj1 = assemble(five_qc)
        state1 = sim.run(qobj1).result().get_statevector()

        #encoding of five qubits
        five_qc.z(4)
        five_qc.h(4)
        five_qc.z(4)
        five_qc.cx(4, 3)
        five_qc.h(3)
        five_qc.h(4)
        five_qc.cx(4, 2)
        five_qc.cx(3, 2)
        five_qc.h(2)
        five_qc.cx(4, 1)
        five_qc.cx(2, 1)
        five_qc.h(4)
        five_qc.h(1)
        five_qc.cx(4, 0)
        five_qc.cx(3, 0)
        five_qc.h(4)
        five_qc.cx(2, 0)
        five_qc.h(3)
        #five_qc.barrier([0,1,2,3,4,5,6,7,8])


        #Error on encoded qubits
        bit = random.randint(0,4) #bit to apply error to
        prob = random.choices([0,1], weights=[(10-2)/10, 2/10]) #probability of an error on encoded qubits
        if prob[0] == 1:
            phase_flip(five_qc, bit)
            five_qc.barrier([qr2,qr1])
        else:
            five_qc.barrier([qr2,qr1])


        #this is the five qubit stabilizer (4-8) with ancilla measurements (0-3) target on codewords
        five_qc.h(5)
        five_qc.h(6)
        five_qc.h(7)
        five_qc.h(8)
        five_qc.cz(5, 1)
        five_qc.cx(5, 2)
        five_qc.cx(5, 3)
        five_qc.cz(5, 4)
        five_qc.cz(6, 0)
        five_qc.cz(6, 2)
        five_qc.cx(6, 3)
        five_qc.cx(6, 4)
        five_qc.cx(7, 0)
        five_qc.cz(7, 1)
        five_qc.cz(7, 3)
        five_qc.cx(7, 4)
        five_qc.cx(8, 0)
        five_qc.cx(8, 1)
        five_qc.cz(8, 2)
        five_qc.cz(8, 4)
        five_qc.h(5)
        five_qc.h(6)
        five_qc.h(7)
        five_qc.h(8)
        five_qc.measure(5,0)
        five_qc.measure(6,1)
        five_qc.measure(7,2)
        five_qc.measure(8,3)
        
        #check ancilla measurements
        qobj = assemble(five_qc)
        results = sim.run(qobj).result()

        #do error correction
        five_qc.x(1).c_if(cr, 1)
        five_qc.z(3).c_if(cr, 2)
        five_qc.x(0).c_if(cr, 3)
        five_qc.z(0).c_if(cr, 4)
        five_qc.z(2).c_if(cr, 5)
        five_qc.x(4).c_if(cr, 6)
        five_qc.y(0).c_if(cr, 7)
        five_qc.x(2).c_if(cr, 8)
        five_qc.z(4).c_if(cr, 9)
        five_qc.z(1).c_if(cr, 10)
        five_qc.y(1).c_if(cr, 11)
        five_qc.x(3).c_if(cr, 12)
        five_qc.y(2).c_if(cr, 13)
        five_qc.y(3).c_if(cr, 14)
        five_qc.y(4).c_if(cr, 15)
        

        #decode the five_qc
        #five_qc.barrier([0,1,2,3,4,5,6,7,8])
        five_qc.h(4)
        five_qc.cx(2, 0)
        five_qc.h(3)
        five_qc.cx(3, 0)
        five_qc.cx(4, 0)
        five_qc.h(1)
        five_qc.cx(2, 1)
        five_qc.h(4)
        five_qc.cx(4, 1)
        five_qc.h(2)
        five_qc.cx(3, 2)
        five_qc.cx(4, 2)
        five_qc.h(4)
        five_qc.h(3)
        five_qc.cx(4, 3)
        five_qc.z(4)
        five_qc.h(4)
        five_qc.z(4)

        #run five_qc
        qobj2 = assemble(five_qc)
        state2 = sim.run(qobj2).result().get_statevector() #state after time step to compare fidelity
        fid = state_fidelity(state1,state2)
        five_fidelity += fid
        avg_fid = five_fidelity/shots
    
    five_array.append(avg_fid)
    #simulator
    sim = Aer.get_backend('statevector_simulator')

    #inital quantum state
    qr1 = QuantumRegister(4, 'ancilla')
    qr2 = QuantumRegister(5, 'q')
    cr = ClassicalRegister(4, 'c')
    five_qc = QuantumCircuit(qr2, qr1, cr)

    five_fidelity = 0.0
    avg_fid = 0.0

    #qobj1 = assemble(five_qc)
    #state1 = sim.run(qobj1).result().get_statevector()

    for i in range(0, shots):
        qobj1 = assemble(five_qc)
        state1 = sim.run(qobj1).result().get_statevector()

        #encoding of five qubits
        five_qc.z(4)
        five_qc.h(4)
        five_qc.z(4)
        five_qc.cx(4, 3)
        five_qc.h(3)
        five_qc.h(4)
        five_qc.cx(4, 2)
        five_qc.cx(3, 2)
        five_qc.h(2)
        five_qc.cx(4, 1)
        five_qc.cx(2, 1)
        five_qc.h(4)
        five_qc.h(1)
        five_qc.cx(4, 0)
        five_qc.cx(3, 0)
        five_qc.h(4)
        five_qc.cx(2, 0)
        five_qc.h(3)
        #five_qc.barrier([0,1,2,3,4,5,6,7,8])


        #Error on encoded qubits
        bit = random.randint(0,4) #bit to apply error to
        prob = random.choices([0,1], weights=[(10-3)/10, 3/10]) #probability of an error on encoded qubits
        if prob[0] == 1:
            phase_flip(five_qc, bit)
            five_qc.barrier([qr2,qr1])
        else:
            five_qc.barrier([qr2,qr1])


        #this is the five qubit stabilizer (4-8) with ancilla measurements (0-3) target on codewords
        five_qc.h(5)
        five_qc.h(6)
        five_qc.h(7)
        five_qc.h(8)
        five_qc.cz(5, 1)
        five_qc.cx(5, 2)
        five_qc.cx(5, 3)
        five_qc.cz(5, 4)
        five_qc.cz(6, 0)
        five_qc.cz(6, 2)
        five_qc.cx(6, 3)
        five_qc.cx(6, 4)
        five_qc.cx(7, 0)
        five_qc.cz(7, 1)
        five_qc.cz(7, 3)
        five_qc.cx(7, 4)
        five_qc.cx(8, 0)
        five_qc.cx(8, 1)
        five_qc.cz(8, 2)
        five_qc.cz(8, 4)
        five_qc.h(5)
        five_qc.h(6)
        five_qc.h(7)
        five_qc.h(8)
        five_qc.measure(5,0)
        five_qc.measure(6,1)
        five_qc.measure(7,2)
        five_qc.measure(8,3)
        
        #check ancilla measurements
        qobj = assemble(five_qc)
        results = sim.run(qobj).result()

        #do error correction
        five_qc.x(1).c_if(cr, 1)
        five_qc.z(3).c_if(cr, 2)
        five_qc.x(0).c_if(cr, 3)
        five_qc.z(0).c_if(cr, 4)
        five_qc.z(2).c_if(cr, 5)
        five_qc.x(4).c_if(cr, 6)
        five_qc.y(0).c_if(cr, 7)
        five_qc.x(2).c_if(cr, 8)
        five_qc.z(4).c_if(cr, 9)
        five_qc.z(1).c_if(cr, 10)
        five_qc.y(1).c_if(cr, 11)
        five_qc.x(3).c_if(cr, 12)
        five_qc.y(2).c_if(cr, 13)
        five_qc.y(3).c_if(cr, 14)
        five_qc.y(4).c_if(cr, 15)
        

        #decode the five_qc
        #five_qc.barrier([0,1,2,3,4,5,6,7,8])
        five_qc.h(4)
        five_qc.cx(2, 0)
        five_qc.h(3)
        five_qc.cx(3, 0)
        five_qc.cx(4, 0)
        five_qc.h(1)
        five_qc.cx(2, 1)
        five_qc.h(4)
        five_qc.cx(4, 1)
        five_qc.h(2)
        five_qc.cx(3, 2)
        five_qc.cx(4, 2)
        five_qc.h(4)
        five_qc.h(3)
        five_qc.cx(4, 3)
        five_qc.z(4)
        five_qc.h(4)
        five_qc.z(4)

        #run five_qc
        qobj2 = assemble(five_qc)
        state2 = sim.run(qobj2).result().get_statevector() #state after time step to compare fidelity
        fid = state_fidelity(state1,state2)
        five_fidelity += fid
        avg_fid = five_fidelity/shots
    
    five_array.append(avg_fid)
    #simulator
    sim = Aer.get_backend('statevector_simulator')

    #inital quantum state
    qr1 = QuantumRegister(4, 'ancilla')
    qr2 = QuantumRegister(5, 'q')
    cr = ClassicalRegister(4, 'c')
    five_qc = QuantumCircuit(qr2, qr1, cr)

    five_fidelity = 0.0
    avg_fid = 0.0

    #qobj1 = assemble(five_qc)
    #state1 = sim.run(qobj1).result().get_statevector()

    for i in range(0, shots):
        qobj1 = assemble(five_qc)
        state1 = sim.run(qobj1).result().get_statevector()

        #encoding of five qubits
        five_qc.z(4)
        five_qc.h(4)
        five_qc.z(4)
        five_qc.cx(4, 3)
        five_qc.h(3)
        five_qc.h(4)
        five_qc.cx(4, 2)
        five_qc.cx(3, 2)
        five_qc.h(2)
        five_qc.cx(4, 1)
        five_qc.cx(2, 1)
        five_qc.h(4)
        five_qc.h(1)
        five_qc.cx(4, 0)
        five_qc.cx(3, 0)
        five_qc.h(4)
        five_qc.cx(2, 0)
        five_qc.h(3)
        #five_qc.barrier([0,1,2,3,4,5,6,7,8])


        #Error on encoded qubits
        bit = random.randint(0,4) #bit to apply error to
        prob = random.choices([0,1], weights=[(10-4)/10, 4/10]) #probability of an error on encoded qubits
        if prob[0] == 1:
            phase_flip(five_qc, bit)
            five_qc.barrier([qr2,qr1])
        else:
            five_qc.barrier([qr2,qr1])


        #this is the five qubit stabilizer (4-8) with ancilla measurements (0-3) target on codewords
        five_qc.h(5)
        five_qc.h(6)
        five_qc.h(7)
        five_qc.h(8)
        five_qc.cz(5, 1)
        five_qc.cx(5, 2)
        five_qc.cx(5, 3)
        five_qc.cz(5, 4)
        five_qc.cz(6, 0)
        five_qc.cz(6, 2)
        five_qc.cx(6, 3)
        five_qc.cx(6, 4)
        five_qc.cx(7, 0)
        five_qc.cz(7, 1)
        five_qc.cz(7, 3)
        five_qc.cx(7, 4)
        five_qc.cx(8, 0)
        five_qc.cx(8, 1)
        five_qc.cz(8, 2)
        five_qc.cz(8, 4)
        five_qc.h(5)
        five_qc.h(6)
        five_qc.h(7)
        five_qc.h(8)
        five_qc.measure(5,0)
        five_qc.measure(6,1)
        five_qc.measure(7,2)
        five_qc.measure(8,3)
        
        #check ancilla measurements
        qobj = assemble(five_qc)
        results = sim.run(qobj).result()

        #do error correction
        five_qc.x(1).c_if(cr, 1)
        five_qc.z(3).c_if(cr, 2)
        five_qc.x(0).c_if(cr, 3)
        five_qc.z(0).c_if(cr, 4)
        five_qc.z(2).c_if(cr, 5)
        five_qc.x(4).c_if(cr, 6)
        five_qc.y(0).c_if(cr, 7)
        five_qc.x(2).c_if(cr, 8)
        five_qc.z(4).c_if(cr, 9)
        five_qc.z(1).c_if(cr, 10)
        five_qc.y(1).c_if(cr, 11)
        five_qc.x(3).c_if(cr, 12)
        five_qc.y(2).c_if(cr, 13)
        five_qc.y(3).c_if(cr, 14)
        five_qc.y(4).c_if(cr, 15)
        

        #decode the five_qc
        #five_qc.barrier([0,1,2,3,4,5,6,7,8])
        five_qc.h(4)
        five_qc.cx(2, 0)
        five_qc.h(3)
        five_qc.cx(3, 0)
        five_qc.cx(4, 0)
        five_qc.h(1)
        five_qc.cx(2, 1)
        five_qc.h(4)
        five_qc.cx(4, 1)
        five_qc.h(2)
        five_qc.cx(3, 2)
        five_qc.cx(4, 2)
        five_qc.h(4)
        five_qc.h(3)
        five_qc.cx(4, 3)
        five_qc.z(4)
        five_qc.h(4)
        five_qc.z(4)

        #run five_qc
        qobj2 = assemble(five_qc)
        state2 = sim.run(qobj2).result().get_statevector() #state after time step to compare fidelity
        fid = state_fidelity(state1,state2)
        five_fidelity += fid
        avg_fid = five_fidelity/shots
    
    five_array.append(avg_fid)
    #simulator
    sim = Aer.get_backend('statevector_simulator')

    #inital quantum state
    qr1 = QuantumRegister(4, 'ancilla')
    qr2 = QuantumRegister(5, 'q')
    cr = ClassicalRegister(4, 'c')
    five_qc = QuantumCircuit(qr2, qr1, cr)

    five_fidelity = 0.0
    avg_fid = 0.0

    #qobj1 = assemble(five_qc)
    #state1 = sim.run(qobj1).result().get_statevector()

    for i in range(0, shots):
        qobj1 = assemble(five_qc)
        state1 = sim.run(qobj1).result().get_statevector()

        #encoding of five qubits
        five_qc.z(4)
        five_qc.h(4)
        five_qc.z(4)
        five_qc.cx(4, 3)
        five_qc.h(3)
        five_qc.h(4)
        five_qc.cx(4, 2)
        five_qc.cx(3, 2)
        five_qc.h(2)
        five_qc.cx(4, 1)
        five_qc.cx(2, 1)
        five_qc.h(4)
        five_qc.h(1)
        five_qc.cx(4, 0)
        five_qc.cx(3, 0)
        five_qc.h(4)
        five_qc.cx(2, 0)
        five_qc.h(3)
        #five_qc.barrier([0,1,2,3,4,5,6,7,8])


        #Error on encoded qubits
        bit = random.randint(0,4) #bit to apply error to
        prob = random.choices([0,1], weights=[(10-5)/10, 5/10]) #probability of an error on encoded qubits
        if prob[0] == 1:
            phase_flip(five_qc, bit)
            five_qc.barrier([qr2,qr1])
        else:
            five_qc.barrier([qr2,qr1])


        #this is the five qubit stabilizer (4-8) with ancilla measurements (0-3) target on codewords
        five_qc.h(5)
        five_qc.h(6)
        five_qc.h(7)
        five_qc.h(8)
        five_qc.cz(5, 1)
        five_qc.cx(5, 2)
        five_qc.cx(5, 3)
        five_qc.cz(5, 4)
        five_qc.cz(6, 0)
        five_qc.cz(6, 2)
        five_qc.cx(6, 3)
        five_qc.cx(6, 4)
        five_qc.cx(7, 0)
        five_qc.cz(7, 1)
        five_qc.cz(7, 3)
        five_qc.cx(7, 4)
        five_qc.cx(8, 0)
        five_qc.cx(8, 1)
        five_qc.cz(8, 2)
        five_qc.cz(8, 4)
        five_qc.h(5)
        five_qc.h(6)
        five_qc.h(7)
        five_qc.h(8)
        five_qc.measure(5,0)
        five_qc.measure(6,1)
        five_qc.measure(7,2)
        five_qc.measure(8,3)
        
        #check ancilla measurements
        qobj = assemble(five_qc)
        results = sim.run(qobj).result()

        #do error correction
        five_qc.x(1).c_if(cr, 1)
        five_qc.z(3).c_if(cr, 2)
        five_qc.x(0).c_if(cr, 3)
        five_qc.z(0).c_if(cr, 4)
        five_qc.z(2).c_if(cr, 5)
        five_qc.x(4).c_if(cr, 6)
        five_qc.y(0).c_if(cr, 7)
        five_qc.x(2).c_if(cr, 8)
        five_qc.z(4).c_if(cr, 9)
        five_qc.z(1).c_if(cr, 10)
        five_qc.y(1).c_if(cr, 11)
        five_qc.x(3).c_if(cr, 12)
        five_qc.y(2).c_if(cr, 13)
        five_qc.y(3).c_if(cr, 14)
        five_qc.y(4).c_if(cr, 15)
        

        #decode the five_qc
        #five_qc.barrier([0,1,2,3,4,5,6,7,8])
        five_qc.h(4)
        five_qc.cx(2, 0)
        five_qc.h(3)
        five_qc.cx(3, 0)
        five_qc.cx(4, 0)
        five_qc.h(1)
        five_qc.cx(2, 1)
        five_qc.h(4)
        five_qc.cx(4, 1)
        five_qc.h(2)
        five_qc.cx(3, 2)
        five_qc.cx(4, 2)
        five_qc.h(4)
        five_qc.h(3)
        five_qc.cx(4, 3)
        five_qc.z(4)
        five_qc.h(4)
        five_qc.z(4)

        #run five_qc
        qobj2 = assemble(five_qc)
        state2 = sim.run(qobj2).result().get_statevector() #state after time step to compare fidelity
        fid = state_fidelity(state1,state2)
        five_fidelity += fid
        avg_fid = five_fidelity/shots
    
    five_array.append(avg_fid)
    #simulator
    sim = Aer.get_backend('statevector_simulator')

    #inital quantum state
    qr1 = QuantumRegister(4, 'ancilla')
    qr2 = QuantumRegister(5, 'q')
    cr = ClassicalRegister(4, 'c')
    five_qc = QuantumCircuit(qr2, qr1, cr)

    five_fidelity = 0.0
    avg_fid = 0.0

    #qobj1 = assemble(five_qc)
    #state1 = sim.run(qobj1).result().get_statevector()

    for i in range(0, shots):
        qobj1 = assemble(five_qc)
        state1 = sim.run(qobj1).result().get_statevector()

        #encoding of five qubits
        five_qc.z(4)
        five_qc.h(4)
        five_qc.z(4)
        five_qc.cx(4, 3)
        five_qc.h(3)
        five_qc.h(4)
        five_qc.cx(4, 2)
        five_qc.cx(3, 2)
        five_qc.h(2)
        five_qc.cx(4, 1)
        five_qc.cx(2, 1)
        five_qc.h(4)
        five_qc.h(1)
        five_qc.cx(4, 0)
        five_qc.cx(3, 0)
        five_qc.h(4)
        five_qc.cx(2, 0)
        five_qc.h(3)
        #five_qc.barrier([0,1,2,3,4,5,6,7,8])


        #Error on encoded qubits
        bit = random.randint(0,4) #bit to apply error to
        prob = random.choices([0,1], weights=[(10-6)/10, 6/10]) #probability of an error on encoded qubits
        if prob[0] == 1:
            phase_flip(five_qc, bit)
            five_qc.barrier([qr2,qr1])
        else:
            five_qc.barrier([qr2,qr1])


        #this is the five qubit stabilizer (4-8) with ancilla measurements (0-3) target on codewords
        five_qc.h(5)
        five_qc.h(6)
        five_qc.h(7)
        five_qc.h(8)
        five_qc.cz(5, 1)
        five_qc.cx(5, 2)
        five_qc.cx(5, 3)
        five_qc.cz(5, 4)
        five_qc.cz(6, 0)
        five_qc.cz(6, 2)
        five_qc.cx(6, 3)
        five_qc.cx(6, 4)
        five_qc.cx(7, 0)
        five_qc.cz(7, 1)
        five_qc.cz(7, 3)
        five_qc.cx(7, 4)
        five_qc.cx(8, 0)
        five_qc.cx(8, 1)
        five_qc.cz(8, 2)
        five_qc.cz(8, 4)
        five_qc.h(5)
        five_qc.h(6)
        five_qc.h(7)
        five_qc.h(8)
        five_qc.measure(5,0)
        five_qc.measure(6,1)
        five_qc.measure(7,2)
        five_qc.measure(8,3)
        
        #check ancilla measurements
        qobj = assemble(five_qc)
        results = sim.run(qobj).result()

        #do error correction
        five_qc.x(1).c_if(cr, 1)
        five_qc.z(3).c_if(cr, 2)
        five_qc.x(0).c_if(cr, 3)
        five_qc.z(0).c_if(cr, 4)
        five_qc.z(2).c_if(cr, 5)
        five_qc.x(4).c_if(cr, 6)
        five_qc.y(0).c_if(cr, 7)
        five_qc.x(2).c_if(cr, 8)
        five_qc.z(4).c_if(cr, 9)
        five_qc.z(1).c_if(cr, 10)
        five_qc.y(1).c_if(cr, 11)
        five_qc.x(3).c_if(cr, 12)
        five_qc.y(2).c_if(cr, 13)
        five_qc.y(3).c_if(cr, 14)
        five_qc.y(4).c_if(cr, 15)
        

        #decode the five_qc
        #five_qc.barrier([0,1,2,3,4,5,6,7,8])
        five_qc.h(4)
        five_qc.cx(2, 0)
        five_qc.h(3)
        five_qc.cx(3, 0)
        five_qc.cx(4, 0)
        five_qc.h(1)
        five_qc.cx(2, 1)
        five_qc.h(4)
        five_qc.cx(4, 1)
        five_qc.h(2)
        five_qc.cx(3, 2)
        five_qc.cx(4, 2)
        five_qc.h(4)
        five_qc.h(3)
        five_qc.cx(4, 3)
        five_qc.z(4)
        five_qc.h(4)
        five_qc.z(4)

        #run five_qc
        qobj2 = assemble(five_qc)
        state2 = sim.run(qobj2).result().get_statevector() #state after time step to compare fidelity
        fid = state_fidelity(state1,state2)
        five_fidelity += fid
        avg_fid = five_fidelity/shots
    
    five_array.append(avg_fid)
    #simulator
    sim = Aer.get_backend('statevector_simulator')

    #inital quantum state
    qr1 = QuantumRegister(4, 'ancilla')
    qr2 = QuantumRegister(5, 'q')
    cr = ClassicalRegister(4, 'c')
    five_qc = QuantumCircuit(qr2, qr1, cr)

    five_fidelity = 0.0
    avg_fid = 0.0

    #qobj1 = assemble(five_qc)
    #state1 = sim.run(qobj1).result().get_statevector()

    for i in range(0, shots):
        qobj1 = assemble(five_qc)
        state1 = sim.run(qobj1).result().get_statevector()

        #encoding of five qubits
        five_qc.z(4)
        five_qc.h(4)
        five_qc.z(4)
        five_qc.cx(4, 3)
        five_qc.h(3)
        five_qc.h(4)
        five_qc.cx(4, 2)
        five_qc.cx(3, 2)
        five_qc.h(2)
        five_qc.cx(4, 1)
        five_qc.cx(2, 1)
        five_qc.h(4)
        five_qc.h(1)
        five_qc.cx(4, 0)
        five_qc.cx(3, 0)
        five_qc.h(4)
        five_qc.cx(2, 0)
        five_qc.h(3)
        #five_qc.barrier([0,1,2,3,4,5,6,7,8])


        #Error on encoded qubits
        bit = random.randint(0,4) #bit to apply error to
        prob = random.choices([0,1], weights=[(10-7)/10, 7/10]) #probability of an error on encoded qubits
        if prob[0] == 1:
            phase_flip(five_qc, bit)
            five_qc.barrier([qr2,qr1])
        else:
            five_qc.barrier([qr2,qr1])


        #this is the five qubit stabilizer (4-8) with ancilla measurements (0-3) target on codewords
        five_qc.h(5)
        five_qc.h(6)
        five_qc.h(7)
        five_qc.h(8)
        five_qc.cz(5, 1)
        five_qc.cx(5, 2)
        five_qc.cx(5, 3)
        five_qc.cz(5, 4)
        five_qc.cz(6, 0)
        five_qc.cz(6, 2)
        five_qc.cx(6, 3)
        five_qc.cx(6, 4)
        five_qc.cx(7, 0)
        five_qc.cz(7, 1)
        five_qc.cz(7, 3)
        five_qc.cx(7, 4)
        five_qc.cx(8, 0)
        five_qc.cx(8, 1)
        five_qc.cz(8, 2)
        five_qc.cz(8, 4)
        five_qc.h(5)
        five_qc.h(6)
        five_qc.h(7)
        five_qc.h(8)
        five_qc.measure(5,0)
        five_qc.measure(6,1)
        five_qc.measure(7,2)
        five_qc.measure(8,3)
        
        #check ancilla measurements
        qobj = assemble(five_qc)
        results = sim.run(qobj).result()

        #do error correction
        five_qc.x(1).c_if(cr, 1)
        five_qc.z(3).c_if(cr, 2)
        five_qc.x(0).c_if(cr, 3)
        five_qc.z(0).c_if(cr, 4)
        five_qc.z(2).c_if(cr, 5)
        five_qc.x(4).c_if(cr, 6)
        five_qc.y(0).c_if(cr, 7)
        five_qc.x(2).c_if(cr, 8)
        five_qc.z(4).c_if(cr, 9)
        five_qc.z(1).c_if(cr, 10)
        five_qc.y(1).c_if(cr, 11)
        five_qc.x(3).c_if(cr, 12)
        five_qc.y(2).c_if(cr, 13)
        five_qc.y(3).c_if(cr, 14)
        five_qc.y(4).c_if(cr, 15)
        

        #decode the five_qc
        #five_qc.barrier([0,1,2,3,4,5,6,7,8])
        five_qc.h(4)
        five_qc.cx(2, 0)
        five_qc.h(3)
        five_qc.cx(3, 0)
        five_qc.cx(4, 0)
        five_qc.h(1)
        five_qc.cx(2, 1)
        five_qc.h(4)
        five_qc.cx(4, 1)
        five_qc.h(2)
        five_qc.cx(3, 2)
        five_qc.cx(4, 2)
        five_qc.h(4)
        five_qc.h(3)
        five_qc.cx(4, 3)
        five_qc.z(4)
        five_qc.h(4)
        five_qc.z(4)

        #run five_qc
        qobj2 = assemble(five_qc)
        state2 = sim.run(qobj2).result().get_statevector() #state after time step to compare fidelity
        fid = state_fidelity(state1,state2)
        five_fidelity += fid
        avg_fid = five_fidelity/shots
    
    five_array.append(avg_fid)
    #simulator
    sim = Aer.get_backend('statevector_simulator')

    #inital quantum state
    qr1 = QuantumRegister(4, 'ancilla')
    qr2 = QuantumRegister(5, 'q')
    cr = ClassicalRegister(4, 'c')
    five_qc = QuantumCircuit(qr2, qr1, cr)

    five_fidelity = 0.0
    avg_fid = 0.0

    #qobj1 = assemble(five_qc)
    #state1 = sim.run(qobj1).result().get_statevector()

    for i in range(0, shots):
        qobj1 = assemble(five_qc)
        state1 = sim.run(qobj1).result().get_statevector()

        #encoding of five qubits
        five_qc.z(4)
        five_qc.h(4)
        five_qc.z(4)
        five_qc.cx(4, 3)
        five_qc.h(3)
        five_qc.h(4)
        five_qc.cx(4, 2)
        five_qc.cx(3, 2)
        five_qc.h(2)
        five_qc.cx(4, 1)
        five_qc.cx(2, 1)
        five_qc.h(4)
        five_qc.h(1)
        five_qc.cx(4, 0)
        five_qc.cx(3, 0)
        five_qc.h(4)
        five_qc.cx(2, 0)
        five_qc.h(3)
        #five_qc.barrier([0,1,2,3,4,5,6,7,8])


        #Error on encoded qubits
        bit = random.randint(0,4) #bit to apply error to
        prob = random.choices([0,1], weights=[(10-8)/10, 8/10]) #probability of an error on encoded qubits
        if prob[0] == 1:
            phase_flip(five_qc, bit)
            five_qc.barrier([qr2,qr1])
        else:
            five_qc.barrier([qr2,qr1])


        #this is the five qubit stabilizer (4-8) with ancilla measurements (0-3) target on codewords
        five_qc.h(5)
        five_qc.h(6)
        five_qc.h(7)
        five_qc.h(8)
        five_qc.cz(5, 1)
        five_qc.cx(5, 2)
        five_qc.cx(5, 3)
        five_qc.cz(5, 4)
        five_qc.cz(6, 0)
        five_qc.cz(6, 2)
        five_qc.cx(6, 3)
        five_qc.cx(6, 4)
        five_qc.cx(7, 0)
        five_qc.cz(7, 1)
        five_qc.cz(7, 3)
        five_qc.cx(7, 4)
        five_qc.cx(8, 0)
        five_qc.cx(8, 1)
        five_qc.cz(8, 2)
        five_qc.cz(8, 4)
        five_qc.h(5)
        five_qc.h(6)
        five_qc.h(7)
        five_qc.h(8)
        five_qc.measure(5,0)
        five_qc.measure(6,1)
        five_qc.measure(7,2)
        five_qc.measure(8,3)
        
        #check ancilla measurements
        qobj = assemble(five_qc)
        results = sim.run(qobj).result()

        #do error correction
        five_qc.x(1).c_if(cr, 1)
        five_qc.z(3).c_if(cr, 2)
        five_qc.x(0).c_if(cr, 3)
        five_qc.z(0).c_if(cr, 4)
        five_qc.z(2).c_if(cr, 5)
        five_qc.x(4).c_if(cr, 6)
        five_qc.y(0).c_if(cr, 7)
        five_qc.x(2).c_if(cr, 8)
        five_qc.z(4).c_if(cr, 9)
        five_qc.z(1).c_if(cr, 10)
        five_qc.y(1).c_if(cr, 11)
        five_qc.x(3).c_if(cr, 12)
        five_qc.y(2).c_if(cr, 13)
        five_qc.y(3).c_if(cr, 14)
        five_qc.y(4).c_if(cr, 15)
        

        #decode the five_qc
        #five_qc.barrier([0,1,2,3,4,5,6,7,8])
        five_qc.h(4)
        five_qc.cx(2, 0)
        five_qc.h(3)
        five_qc.cx(3, 0)
        five_qc.cx(4, 0)
        five_qc.h(1)
        five_qc.cx(2, 1)
        five_qc.h(4)
        five_qc.cx(4, 1)
        five_qc.h(2)
        five_qc.cx(3, 2)
        five_qc.cx(4, 2)
        five_qc.h(4)
        five_qc.h(3)
        five_qc.cx(4, 3)
        five_qc.z(4)
        five_qc.h(4)
        five_qc.z(4)

        #run five_qc
        qobj2 = assemble(five_qc)
        state2 = sim.run(qobj2).result().get_statevector() #state after time step to compare fidelity
        fid = state_fidelity(state1,state2)
        five_fidelity += fid
        avg_fid = five_fidelity/shots
    
    five_array.append(avg_fid)
    #simulator
    sim = Aer.get_backend('statevector_simulator')

    #inital quantum state
    qr1 = QuantumRegister(4, 'ancilla')
    qr2 = QuantumRegister(5, 'q')
    cr = ClassicalRegister(4, 'c')
    five_qc = QuantumCircuit(qr2, qr1, cr)

    five_fidelity = 0.0
    avg_fid = 0.0

    #qobj1 = assemble(five_qc)
    #state1 = sim.run(qobj1).result().get_statevector()

    for i in range(0, shots):
        qobj1 = assemble(five_qc)
        state1 = sim.run(qobj1).result().get_statevector()

        #encoding of five qubits
        five_qc.z(4)
        five_qc.h(4)
        five_qc.z(4)
        five_qc.cx(4, 3)
        five_qc.h(3)
        five_qc.h(4)
        five_qc.cx(4, 2)
        five_qc.cx(3, 2)
        five_qc.h(2)
        five_qc.cx(4, 1)
        five_qc.cx(2, 1)
        five_qc.h(4)
        five_qc.h(1)
        five_qc.cx(4, 0)
        five_qc.cx(3, 0)
        five_qc.h(4)
        five_qc.cx(2, 0)
        five_qc.h(3)
        #five_qc.barrier([0,1,2,3,4,5,6,7,8])


        #Error on encoded qubits
        bit = random.randint(0,4) #bit to apply error to
        prob = random.choices([0,1], weights=[(10-9)/10, 9/10]) #probability of an error on encoded qubits
        if prob[0] == 1:
            phase_flip(five_qc, bit)
            five_qc.barrier([qr2,qr1])
        else:
            five_qc.barrier([qr2,qr1])


        #this is the five qubit stabilizer (4-8) with ancilla measurements (0-3) target on codewords
        five_qc.h(5)
        five_qc.h(6)
        five_qc.h(7)
        five_qc.h(8)
        five_qc.cz(5, 1)
        five_qc.cx(5, 2)
        five_qc.cx(5, 3)
        five_qc.cz(5, 4)
        five_qc.cz(6, 0)
        five_qc.cz(6, 2)
        five_qc.cx(6, 3)
        five_qc.cx(6, 4)
        five_qc.cx(7, 0)
        five_qc.cz(7, 1)
        five_qc.cz(7, 3)
        five_qc.cx(7, 4)
        five_qc.cx(8, 0)
        five_qc.cx(8, 1)
        five_qc.cz(8, 2)
        five_qc.cz(8, 4)
        five_qc.h(5)
        five_qc.h(6)
        five_qc.h(7)
        five_qc.h(8)
        five_qc.measure(5,0)
        five_qc.measure(6,1)
        five_qc.measure(7,2)
        five_qc.measure(8,3)
        
        #check ancilla measurements
        qobj = assemble(five_qc)
        results = sim.run(qobj).result()

        #do error correction
        five_qc.x(1).c_if(cr, 1)
        five_qc.z(3).c_if(cr, 2)
        five_qc.x(0).c_if(cr, 3)
        five_qc.z(0).c_if(cr, 4)
        five_qc.z(2).c_if(cr, 5)
        five_qc.x(4).c_if(cr, 6)
        five_qc.y(0).c_if(cr, 7)
        five_qc.x(2).c_if(cr, 8)
        five_qc.z(4).c_if(cr, 9)
        five_qc.z(1).c_if(cr, 10)
        five_qc.y(1).c_if(cr, 11)
        five_qc.x(3).c_if(cr, 12)
        five_qc.y(2).c_if(cr, 13)
        five_qc.y(3).c_if(cr, 14)
        five_qc.y(4).c_if(cr, 15)
        

        #decode the five_qc
        #five_qc.barrier([0,1,2,3,4,5,6,7,8])
        five_qc.h(4)
        five_qc.cx(2, 0)
        five_qc.h(3)
        five_qc.cx(3, 0)
        five_qc.cx(4, 0)
        five_qc.h(1)
        five_qc.cx(2, 1)
        five_qc.h(4)
        five_qc.cx(4, 1)
        five_qc.h(2)
        five_qc.cx(3, 2)
        five_qc.cx(4, 2)
        five_qc.h(4)
        five_qc.h(3)
        five_qc.cx(4, 3)
        five_qc.z(4)
        five_qc.h(4)
        five_qc.z(4)

        #run five_qc
        qobj2 = assemble(five_qc)
        state2 = sim.run(qobj2).result().get_statevector() #state after time step to compare fidelity
        fid = state_fidelity(state1,state2)
        five_fidelity += fid
        avg_fid = five_fidelity/shots
    
    five_array.append(avg_fid)

    return five_array

five_qubit_stabilizer(1)