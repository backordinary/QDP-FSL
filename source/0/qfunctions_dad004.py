# https://github.com/angel-ibm/motoqp/blob/fbca1e2b18c0b8f118a19234a6ea3dd7027f716b/qfunctions.py
#!/usr/bin/env python3

# These are all the qiskit functions used in the application

# Code written on python 3.9.2, please observe the general compatibility rules

from qiskit import QuantumCircuit, Aer, execute

####################
## TYPE OF RACE ####
##                ##
##      ┌───┐┌─┐  ##
## q_0: ┤ H ├┤M├  ##
##      └───┘└╥┘  ##
## c: 1/══════╩═  ##
##                ##
####################
## This is a trivial random generator: 0 or 1 uniformly distributed
## just one qbit, one hadamard gate, one mesaurement and one shot
def quantum_select_type_race():
    motoqp_runtime = Aer.get_backend('qasm_simulator')
    motoqp_circuit = QuantumCircuit(1, 1)
    motoqp_circuit.h(0)
    motoqp_circuit.measure(0, 0)  # pylint: disable=no-member
    
    motoqp_job = execute(motoqp_circuit, motoqp_runtime, shots=1)
    motoqp_result = motoqp_job.result().data()['counts']
    if ('0x0' in motoqp_result):
        return ('fool')
    else:
        return ('normal')

##########################
##     CIRCUIT TYPE     ##
##      ┌───┐┌─┐        ##
## q_0: ┤ H ├┤M├──────  ##
##      ├───┤└╥┘┌─┐     ##
## q_1: ┤ H ├─╫─┤M├───  ##
##      ├───┤ ║ └╥┘┌─┐  ##
## q_2: ┤ H ├─╫──╫─┤M├  ##
##      └───┘ ║  ║ └╥┘  ##
## c: 3/══════╩══╩══╩═  ##
##            0  1  2   ##
##                      ##
##########################
## This is an easy random generator: 1,2,3,4,5 uniformly distributed
# as it uses 3 qbits, the states 0,6,7 are discarded
def quantum_select_circuit_type():
    motoqp_runtime = Aer.get_backend('qasm_simulator')
    motoqp_circuit = QuantumCircuit(3, 3)
    motoqp_circuit.h(0)
    motoqp_circuit.h(1)
    motoqp_circuit.h(2)
    motoqp_circuit.measure([0, 1, 2], [0, 1, 2])  # pylint: disable=no-member
   
    circuit_type = 'unknown'
    while (circuit_type == 'unknown'):
        motoqp_job = execute(motoqp_circuit, motoqp_runtime, shots=1)
        motoqp_result = motoqp_job.result().data()['counts']
        if ('0x1' in motoqp_result):
            circuit_type = 'straight'
        elif ('0x2' in motoqp_result):
            circuit_type = 'uphills'
        elif ('0x3' in motoqp_result):
            circuit_type = 'downhills'
        elif ('0x4' in motoqp_result):
            circuit_type = 'windy'
        elif ('0x5' in motoqp_result):
            circuit_type = 'rainy'
    return(circuit_type)


################################
##    BIKE ASSIGN SPECS       ##
##                            ##
##      ┌───┐┌─┐              ##
## q_0: ┤ H ├┤M├────────────  ##
##      ├───┤└╥┘┌─┐           ##
## q_1: ┤ H ├─╫─┤M├─────────  ##
##      ├───┤ ║ └╥┘┌─┐        ##
## q_2: ┤ H ├─╫──╫─┤M├──────  ##
##      ├───┤ ║  ║ └╥┘┌─┐     ##
## q_3: ┤ H ├─╫──╫──╫─┤M├───  ##
##      ├───┤ ║  ║  ║ └╥┘┌─┐  ##
## q_4: ┤ H ├─╫──╫──╫──╫─┤M├  ##
##      └───┘ ║  ║  ║  ║ └╥┘  ##
## c: 5/══════╩══╩══╩══╩══╩═  ##
##            0  1  2  3  4   ##
##                            ##
################################
## This is a gaussian-like random generator of 1,2,3,4,5
## 3 has the higher probability, 2 & 4 have lower probability and 1 & 5 a small probability
## it is implemented with 5 qubits and mapping the states for achieving the distribution

def quantum_assign_specs() :

    worst =   ['0x0', '0x1', '0x2' ]
    bad =     ['0x3', '0x4', '0x5', '0x6', '0x7', '0x8', '0x9' ]
    average = ['0xa', '0xb', '0xc', '0xd', '0xe', '0xf', '0x10','0x11','0x12','0x13','0x14','0x15']
    good =    ['0x16','0x17','0x18','0x19','0x1a','0x1b','0x1c']
    best =    ['0x1d','0x1e','0x1f']
    
    motoqp_runtime = Aer.get_backend('qasm_simulator')
    motoqp_circuit = QuantumCircuit(5, 5)
    motoqp_circuit.h(0)
    motoqp_circuit.h(1)
    motoqp_circuit.h(2)
    motoqp_circuit.h(3)
    motoqp_circuit.h(4)
    motoqp_circuit.measure([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])  # pylint: disable=no-member

    motoqp_job = execute(motoqp_circuit, motoqp_runtime, shots=1)
    motoqp_result = str(list(motoqp_job.result().data()['counts'].keys())[0])

    if (motoqp_result in worst):
        return('worst')
    elif (motoqp_result in bad) :
        return('bad')
    elif (motoqp_result in average) :
        return('average')
    elif (motoqp_result in good) :
        return('good')
    elif (motoqp_result in best) :
        return('best')
