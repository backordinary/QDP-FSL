# https://github.com/islegmar/quantumChangePoint/blob/f14e8083254e42dd783eb3c39dd5fe89e98a58f9/code/quantum5.py
# -*- coding: utf-8 -*-
from qiskit import(
  QuantumCircuit,
  QuantumRegister,
  ClassicalRegister,
  execute,
  Aer
)
import math

# TODO : fix the circuits is not reused
def getSuccess(c, experiments, num_iterations_per_experiment=1000):
    """ Execute a serie of experiments.

    experiments is a map { <experiment> : <expected output> } being a serie of
    0s and 1s where:
    - 0 in experiment : state |0>
    - 1 in experiment : state |Phi> (where c = <Phi|0>)
    - 0,1 in expected output : the outputs when measuring in z"""

    theta=math.acos(c)
    simulator = Aer.get_backend('qasm_simulator')
    
    totIterations=0
    totOKs=0
    for experiment, my_output in experiments.items():
        qreg_q = QuantumRegister(5, 'q')
        creg_c = ClassicalRegister(5, 'c')
        circuit = QuantumCircuit(qreg_q, creg_c)
        
        for ind in range(len(experiment)):
            state=experiment[ind]
            # Do nothing, the q-bit is already in state |0>
            if state=='0':
                pass
            # Put it in the state |Phi>
            elif state=='1':
                circuit.initialize([c, math.sqrt(1-c**2)], qreg_q[ind])
        
        # Map the quantum measurement to the classical bits
        circuit.measure(qreg_q, creg_c)
        
        # Execute the circuit on the qasm simulator
        totIterations+=num_iterations_per_experiment
        job = execute(circuit, simulator, shots=num_iterations_per_experiment)
        result = job.result()
        counts = result.get_counts(circuit)
        if my_output in counts:
            totOKs += counts.get(my_output)
    
    return 100.0 * (totOKs/totIterations)

# To suceed, these are the outputs for the experiments, using my 
# estimator function
# key : experiment, value : the measurement I expect to succedd
# NOTE : we have to invert the measurament for the way qiskit returns them
experiments = { '00000' : '00000',
                '00001' : '10000',
                '00011' : '11000',
                '00111' : '11100',
                '01111' : '11110'
              }

print ("c=%f, %f %% " % (1,   getSuccess(1,   experiments)))
print ("c=%f, %f %% " % (0.5, getSuccess(0.5, experiments)))
print ("c=%f, %f %% " % (0,   getSuccess(0,   experiments)))
