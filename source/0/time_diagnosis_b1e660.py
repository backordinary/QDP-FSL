# https://github.com/amarjahin/kitaev_models_vqe/blob/839a1231c8870ab3f4c2015ec3ba6210fc4d0895/time_diagnosis/time_diagnosis.py
import time 
from numpy import pi, conjugate
from numpy.random import random, rand
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import StatevectorSimulator
from qiskit.opflow import PauliOp, PauliSumOp
from qiskit.quantum_info import Pauli
h = PauliOp(Pauli('IIZZIZZZIIZZIZZZ')) + PauliOp(Pauli('IIXXXZZXIIZZIZZZ')) 
h_mat = h.to_spmatrix()
start = time.time()
num_params = 240
num_terms= 2*num_params
params = ParameterVector('a', length = num_params)
qc = QuantumCircuit(16)
# This circuit is not the ansatz, but it looks like it. 
# this construction of the circuit is worst that what it actually is in 
# terms of the number of cx gates.
for i in range(num_terms):
    rz_pos = (i//2) % 14 + 1
    for j in range(rz_pos): 
        qc.cx(j, j+1)
    qc.rz(params[i//2], rz_pos)
    for j in range(rz_pos)[::-1]: 
        qc.cx(j, j+1)
    qc.h([*range(rz_pos)])
qc = transpile(qc)  
end = time.time()
print(f'time took to build the circuit: {round(end - start,3)}')
nfev = 12000 # this is the number of times the optimizer calls the cost function. 
# Just make the calling to the cost function roughly the same times as the optimizer.
simulator = StatevectorSimulator()
bind_params_time = 0 
run_circuit_time = 0 
expectation_time = 0 
for i in range(100): 
    btime = time.time()
    params = pi*random(num_params)
    qc_c = qc.bind_parameters(params)
    bind_params_time = bind_params_time + (time.time()-btime)

    rtime = time.time()
    result = simulator.run(qc_c).result()
    run_circuit_time = run_circuit_time + (time.time()-rtime)

    etime = time.time()
    psi = result.get_statevector()
    e_ev = conjugate(psi.T) @ h_mat @ psi 
    expectation_time = expectation_time + (time.time()-etime)

full_time = time.time() - end            
print(f'time took to call the function {100} times: {round(full_time,3)}s')
print(f'of which time binding parameters: {round(bind_params_time,3)} ({round(bind_params_time/full_time * 100,3)}%)')
print(f'of which time running circuit: {round(run_circuit_time,3)} ({round(run_circuit_time/full_time * 100,3)}%)')
print(f'of which time calculating <H>: {round(expectation_time,3)} ({round(expectation_time /full_time * 100, 3)}%)')
print(f'extrapolating to {10000} calls, it should take {round(full_time * (12000/100) / (60),3)}m')