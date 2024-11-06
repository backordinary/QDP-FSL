# https://github.com/Linueks/QuantumComputing/blob/c5876baad39b9337e7e50549f3f1c7c9d3de53dc/IBM-quantum-challenge/src/mitiq_ZNE_test.py
"""
Trying to learn about Zero Noise Extrapolation using Mitiq from their intro
guide: https://mitiq.readthedocs.io/en/stable/guide/guide-getting-started.html#guide-getting-started
@linueks
"""
import numpy as np
import mitiq as mt
import qiskit as qk
import matplotlib.pyplot as plt
from simulation import test_generate_circuit, run_simulation
from functools import partial


"""
THIS FILE IS JUST LEFT FOR LATER, NOT WORKING NOW.
"""


def circuit_executor(
    circuit,
    backend,
    shots=8192,
    expecation_value_key='110',
):
    """
    Specifically written for how Mitiq works, just to have an example somewhere.
    Use functools.partial in order to pass backend or other variables when using
    mitiq execute_with_zne.

    Inputs:
        circuit: qiskit.circuit.QuantumCircuit
        backend: qiskit.IBMQBackend
        shots: int
        expecation_value_key: string

    return:
        expecation_value: float
    """
    job = qk.execute(
        circuit,
        backend,
        shots=shots
    )
    results = job.result()
    counts = results.get_counts()
    expecation_value = counts[expecation_value_key] / shots

    return expecation_value



provider = qk.IBMQ.load_account()
provider = qk.IBMQ.get_provider(hub='ibm-q', group='open', project='main')
#print(provider.backends())
belem_backend = provider.get_backend('ibmq_belem')                              # has the same topology as Jakarta with qubits 1,3,4 corresponding to 1,3,5
properties = belem_backend.properties()
#print(properties)
config = belem_backend.configuration()
#print(config.backend_name)


seed = 42
qk.utils.algorithm_globals.random_seed = seed


sim_noisy_belem = qk.providers.aer.QasmSimulator.from_backend(belem_backend)
time = qk.circuit.Parameter('t')
shots = 8192
trotter_steps = 7                                                             # Variable if just running one simulation
end_time = np.pi                                                                # Specified in competition
num_jobs = 8


circuit = test_generate_circuit(time,
                                sim_noisy_belem,
                                trotter_steps=trotter_steps,
                                target_time=np.pi,
                                draw_circuit=False)

unmitigated_jobs = run_simulation(circuit,
                                    sim_noisy_belem,
                                    shots=shots,
                                    num_jobs=num_jobs)
print(unmitigated_jobs)

mitigated_jobs = mt.zne.execute_with_zne(circuit,
                                    partial(run_simulation,
                                            backend=sim_noisy_belem,
                                            shots=shots,
                                            num_jobs=num_jobs))
print(mitigated_jobs)


"""
unmitigated_fidelity, std = tomography_analysis(time, unmitigated_jobs, circuit,
                                                register, trotter_steps=steps,
                                                target_time=end_time)
mitigated_fidelity, mitiq_std = tomography_analysis(time, unmitigated_jobs,
                                                circuit, register,
                                                trotter_steps=steps,
                                                target_time=end_time)

print(f'unmitigated fidelity = {unmitigated_fidelity:.4f}',
       '\u00B1', f'{std:.4f}')

print(f'state tomography fidelity = {mitigated_fidelity:.4f}',
       '\u00B1', f'{mitiq_std:.4f}')
"""
