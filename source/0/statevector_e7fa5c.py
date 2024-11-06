# https://github.com/paniash/progs/blob/b7f58efeb5b5c7942af6b8af12611bdbc1a52840/python/qiskit/statevector.py
from qiskit import *
from qiskit.visualization import plot_state_qsphere, plot_histogram
import qiskit.quantum_info as qi

q_sim = Aer.get_backend('qasm_simulator')
s_sim = Aer.get_backend('statevector_simulator')
u_sim = Aer.get_backend('unitary_simulator')

#%%
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0,1)
qc.draw()

#%%
job = execute(qc, s_sim)
vector = job.result().get_statevector()
plot_state_qsphere(vector)

#%%
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0,1)
qc.draw()

#%%
sv = qi.Statevector.from_instruction(qc)
plot_state_qsphere(sv)

#%%
sv2 = qi.Statevector.from_label('11')
sv2 = sv2.evolve(qc)
plot_state_qsphere(sv2)

#%%
state_array = np.array([1/2, -1j/np.sqrt(2), 0, 1/2])
sv3 = qi.Statevector(state_array)
sv3 = sv3.evolve(qc)
plot_state_qsphere(sv3)


#%%
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0,1)
qc.draw()

#%%
probs = sv.probabilities_dict(decimals=6)
for st in probs:
    print("The ideal probability of measuring {} is {}".format(st, probs[st]))

#%%
samples = sv.sample_counts(128)
for st in samples:
    print("State {} was sampled {} times".format(st, samples[st]))

#%%
plot_histogram([probs, samples], title="Ideal vs Sampled probabilities")
