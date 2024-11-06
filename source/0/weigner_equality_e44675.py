# https://github.com/AndreVadi23/prjcqn/blob/166000c87346de750eeb3eaab9740862dc5820b2/Weigner's%20inequality/Weigner_equality.py
from numpy import pi
from qiskit.visualization import plot_histogram
from qiskit_rng import Generator
import matplotlib.pyplot as plt
from qiskit import *
from timeit import default_timer as timer

# TODO
# Exporter et le run en executable et s'assurer que le graph est mis à jour en TR
# Automatiser l'intervention de eve sur une plage d'itérations données

# qbit 0 represents Alice
# qbit 1 represents Bob

ITERATIONS = 10

BATCH_SIZE = 4096
simulator = Aer.get_backend('qasm_simulator')
generator = Generator(backend=simulator)
EVE_INTERVENTION = [4,5,6,7]

#Initialisation,
initialize = QuantumCircuit(2,2)
initialize.x(0)
initialize.h(0)
initialize.x(1)
initialize.cx(0,1)

eve = QuantumCircuit(2,2)
eve.x(0)

# Z Z state
ZZ_state = QuantumCircuit(2,2)
ZZ_state.measure(0,0)
ZZ_state.measure(1,1)
# R Z state
RZ_state = QuantumCircuit(2,2)
RZ_state.ry(-pi/3,0)
RZ_state.measure(0,0)
RZ_state.measure(1,1)
# Z S state
ZS_state = QuantumCircuit(2,2)
ZS_state.ry(pi/3,1)
ZS_state.measure(0,0)
ZS_state.measure(1,1)
# R S state
RS_state = QuantumCircuit(2,2)
RS_state.ry(-pi/3,0)
RS_state.ry(pi/3,1)
RS_state.measure(0,0)
RS_state.measure(1,1)

plt.axis([0, ITERATIONS, -0.3, 0.3])
for i in range(ITERATIONS):
    start = timer()
    key_bits = list()
    counts_rz = list((0,0))
    counts_zs = list((0,0))
    counts_rs = list((0,0))
    for j in range(BATCH_SIZE):
        output = generator.sample(num_raw_bits=1).block_until_ready()
        a = (output.raw_bits[0],output.raw_bits[1])
        if(a == (0,1)):
            if ( i in EVE_INTERVENTION ):
                job = execute(eve + RZ_state, backend = simulator, shots = 1)
            else:
                job = execute(initialize + RZ_state, backend = simulator, shots = 1)
            counts_rz[0] += 1
            results = job.result()
            counts = results.get_counts()
            if '00' in counts:
                counts_rz[1] += 1
        elif(a == (1,0)):
            if ( i in EVE_INTERVENTION ):
                job = execute(eve + ZS_state, backend = simulator, shots = 1)
            else:
                job = execute(initialize + ZS_state, backend = simulator, shots = 1)
            counts_zs[0]+=1 
            results = job.result()
            counts = results.get_counts()
            if '00' in counts:
                counts_zs[1]+=1
        elif(a == (1,1)):
            if ( i in EVE_INTERVENTION ):
                job = execute(eve + RS_state, backend = simulator, shots = 1)
            else:
                job = execute(initialize + RS_state, backend = simulator, shots = 1)
            counts_rs[0]+=1
            results = job.result()
            counts = results.get_counts()
            if '00' in counts:
                counts_rs[1]+=1
        else:
            if ( i in EVE_INTERVENTION ):
                job = execute(eve + ZZ_state, backend = simulator, shots = 1)
            else:
                job = execute(initialize + ZZ_state, backend = simulator, shots = 1)
            results = job.result()
            counts = results.get_counts()
            key_bits.append(list(counts.keys())[0][0])
    key = ''.join(key_bits)
    #W = p_rz + p_zs - p_rs
    if(counts_rz[0] == 0):
        print("Failure")
        p_rz = 0
    else:
        p_rz = counts_rz[1]/counts_rz[0]
    if(counts_zs[0] == 0):
        print("Failure")
        p_zs = 0
    else:
        p_zs = counts_zs[1]/counts_zs[0]
    if(counts_rs[0] == 0):
        print("Failure")
        p_rs = 0
    else:
        p_rs = counts_rs[1]/counts_rs[0]
    W = p_rz + p_zs - p_rs
    end = timer()
    print("Iteration : " + str(i + 1))
    print("P++(-30,0)=" + str(p_rz))
    print("P++(0,30)=" + str(p_zs))
    print("P++(-30,30)=" + str(p_rs))
    print("W=" + str(W))
    print("k=" + str(key))
    print("Time elapsed (seconds) = " + str(end-start))
    plt.scatter(i, W)
    plt.pause(0.5)
# The legend will be displayed only at the end
plt.title("Weigner's equality with listener; BATCH SIZE = " + str(BATCH_SIZE))
plt.xlabel("Number of Iterations")
plt.ylabel("W")
plt.show()
