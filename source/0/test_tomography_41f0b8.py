# https://github.com/wiktor145/QTrator/blob/de8f4ce389bb9b6168096ab615e5b3b8de2b1c11/test_tomography.py
from qiskit import execute, Aer, transpile
from qiskit.ignis.verification import state_tomography_circuits, StateTomographyFitter
from qiskit.providers.aer import QasmSimulator, noise, StatevectorSimulator, AerSimulator
from qiskit.test.mock import FakeProvider, FakeMontreal

provider = FakeProvider()

# for b in provider.backends():
#     print(b.name())
#     print(b.configuration().n_qubits)
# exit(1)

from convert_qasm_to_qiskit import get_circuit_from_qasm_file

backend = QasmSimulator()
backend_hardware = FakeMontreal()

noise_model = noise.NoiseModel.from_backend(backend_hardware)

c = get_circuit_from_qasm_file("/home/wiktor/Documents/mgr_comparator-master/benchmark_files/ham15-med.qasm")

# tomography test
# print(c.qregs)
c.remove_final_measurements()
c.save_statevector()
print(Aer.backends())

#backend = Aer.get_backend('aer_simulator')
# job = execute(c, backend, shots=1)

backend = AerSimulator.from_backend(backend_hardware, method="statevector")
c = transpile(c, backend, optimization_level=0)
job = backend.run(c, shots=1, noise_model=noise_model)
result = job.result()
print(result)
# print(result.get_counts())
print(result.get_statevector())
# print(123)
exit(1)

# print(c)

qregs = []
qregs.append(c.qregs[0][0])
qregs.append(c.qregs[0][1])
qregs.append(c.qregs[0][2])
qregs.append(c.qregs[0][3])
qregs.append(c.qregs[0][4])
qregs.append(c.qregs[0][5])
qregs.append(c.qregs[0][6])
qregs.append(c.qregs[0][7])
qregs.append(c.qregs[0][8])
qregs.append(c.qregs[0][9])

qst_bell = state_tomography_circuits(c, qregs)
print(len(qst_bell))
exit(1)
job = execute(qst_bell, backend, shots=16)
tomo_fitter_bell = StateTomographyFitter(job.result(), qst_bell)
rho_fit_bell = tomo_fitter_bell.fit(method='lstsq')
print(rho_fit_bell)
exit(1)

c = get_circuit_from_qasm_file("/home/wiktor/Documents/mgr_comparator-master/benchmark_files/csum_mux_9.qasm")

qregs = []
qregs.append(c.qregs[0][0])
qregs.append(c.qregs[0][1])
qregs.append(c.qregs[0][2])
qregs.append(c.qregs[0][3])
qregs.append(c.qregs[0][4])
qregs.append(c.qregs[0][5])
qregs.append(c.qregs[0][6])
qregs.append(c.qregs[0][7])
qregs.append(c.qregs[0][8])
qregs.append(c.qregs[0][9])

# tomography test
# print(c.qregs)
c.remove_final_measurements()
# print(c)
qst_bell = state_tomography_circuits(c, qregs)
# print(qst_bell)
job = execute(qst_bell, backend, shots=16)
tomo_fitter_bell = StateTomographyFitter(job.result(), qst_bell)
rho_fit_bell = tomo_fitter_bell.fit(method='lstsq')
print(rho_fit_bell)
