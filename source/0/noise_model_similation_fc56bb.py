# https://github.com/happybored/pennylane_vqe/blob/4723ad0ec40a5cb49a99204d989758d650596780/noise_model_examnation/noise_model_similation.py
from qiskit import IBMQ, transpile,assemble
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator,QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.tools.visualization import plot_histogram
from qiskit.test.mock import FakeVigo
from matplotlib import pyplot as plt
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
from qiskit.ignis.mitigation import MeasurementFilter,CompleteMeasFitter
from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit import Aer,QuantumRegister
from qiskit.aqua.utils.run_circuits import run_qobj
import sys
from qiskit.aqua.utils.measurement_error_mitigation import (get_measured_qubits_from_qobj,build_measurement_error_mitigation_qobj)
import qiskit
device_backend = FakeVigo()
noise_model = NoiseModel.from_backend(device_backend)
coupling_map = device_backend.configuration().coupling_map



# Construct quantum circuit
circ = QuantumCircuit(3, 3)
circ.h(0)
circ.h(2)
circ.cx(0, 1)
circ.cx(1, 2)
circ.ry(0.2,1)
circ.rx(0.2,0)

circ.measure([0, 1, 2], [0, 1, 2])


sim_ideal = AerSimulator()

# Execute and get counts
result = sim_ideal.run(transpile(circ, sim_ideal)).result()
counts = result.get_counts(0)
plt.figure(1)
plot_histogram(counts, title='Ideal counts for 3-qubit GHZ state',filename='hist-ideal.jpg')

# sim_vigo = AerSimulator.from_backend(device_backend)

sim_vigo = AerSimulator(noise_model=noise_model)

#noise backend
# Transpile the circuit for the noisy basis gates



tcirc = transpile(circ, sim_ideal)
qobj = assemble(tcirc, shots=10000)

# Execute noisy simulation and get counts
result_noise = sim_ideal.run(qobj, noise_model=noise_model, shots=10000).result()
counts_noise = result_noise.get_counts()
plt.figure(2)
plot_histogram(counts_noise,
               title="Counts for 3-qubit GHZ state with device noise model",filename='hist-noise.jpg')


# device = QasmSimulator.from_backend(device_backend)
# noise_model = NoiseModel.from_backend(device)
# basis_gates = noise_model.basis_gates
# backend = Aer.get_backend('aer_simulator')
seed =170

qi_noise_mitigation = QuantumInstance(backend=sim_ideal, seed_simulator=seed, seed_transpiler=seed,
                     coupling_map=coupling_map, noise_model=noise_model,
                     measurement_error_mitigation_cls=CompleteMeasFitter,
                     cals_matrix_refresh_period=30)
ret = qi_noise_mitigation.execute(tcirc)
counts = ret.results[0].data.counts
plot_histogram(counts,title="Counts for 3-qubit GHZ state with device noise model and Mitigation",filename='hist-noise-mitigation.jpg')

qr = QuantumRegister(3)
meas_calibs, state_labels = complete_meas_cal(qr=qr,circlabel='mcal')
# for circuit in meas_calibs:
#     print('Circuit',circuit.name)
#     print(circuit)
#     print()

# print(meas_calibs)
t_qc = transpile(meas_calibs, sim_ideal)
qobj = assemble(t_qc, shots=10000)

cal_results = sim_ideal.run(qobj, noise_model=noise_model, shots=10000).result()# print(qobj)

# print(cal_results)
# print()
# print(state_labels)
# print()

meas_fitter = CompleteMeasFitter(cal_results, state_labels,circlabel='mcal')
mitigated_results = meas_fitter.filter.apply(result_noise)

# print(result_noise)
# print()

# print(meas_fitter.cal_matrix)
# print()
# print(mitigated_results)
# print()
mitigated_counts = mitigated_results.get_counts()
plot_histogram(mitigated_counts,title="Counts for 3-qubit GHZ state with device noise model and Mitigation2",filename='hist-noise-mitigation2.jpg')

from mitiq import zne
from mitiq.interface.mitiq_qiskit.qiskit_utils import initialized_depolarizing_noise




shots=10000
circuit =  circ
# Convert from raw measurement counts to the expectation value
scale_factors = [1., 1.5, 2.,2.5,3.0]
folded_circuits = [
        zne.scaling.fold_gates_at_random(circuit, scale)
        for scale in scale_factors
]
job = qiskit.execute(
    experiments=folded_circuits,
    backend=qiskit.Aer.get_backend("qasm_simulator"),
    noise_model=noise_model,
    basis_gates=noise_model.basis_gates,
    optimization_level=0,  # Important to preserve folded gates.
    shots=shots,
)
# Check that the circuit depth is (approximately) scaled as expected
for j, c in enumerate(folded_circuits):
    print(f"Number of gates of folded circuit {j} scaled by: {len(c) / len(circuit):.3f}")
all_counts = [job.result().get_counts(i) for i in range(len(folded_circuits))]
expectation_values = [counts.get("000") / shots for counts in all_counts]
print(expectation_values)



zero_noise_value = zne.ExpFactory.extrapolate(scale_factors, expectation_values, asymptote=0.5)
print(f"Unmitigated result {expectation_values[0]}")
print(f"Mitigated result {zero_noise_value}")