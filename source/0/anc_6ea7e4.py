# https://github.com/Stephen-Campbell-UTD/NM_Project_Quantum_Computing/blob/676e0dfe7f04dfe36e687bb95b5479f752f14218/Qiskit/anc.py
#%%
from qiskit import QuantumCircuit
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import AncillaRegister
# %%
def gen_test_gate() -> Gate:
  active_reg = QuantumCircuit(1)
  anc_reg = AncillaRegister(1)
  active_reg.add_register(anc_reg)
  active_reg.x(0)
  gate = active_reg.to_gate()
  return gate


qc = QuantumCircuit(2)
qc.append(gen_test_gate(), qargs=[0,1])
qc.draw()


# %%
