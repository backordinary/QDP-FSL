# https://github.com/Linueks/QuantumComputing/blob/e85c410a8e9d47fd215b9dbd25a6c5b73daa1f6f/thesis/src/random_circuit_generation.py
import qiskit as qk
import numpy as np
import matplotlib.pyplot as plt
import qiskit.opflow as opflow
import qiskit.ignis.verification.tomography as tomo
from qiskit.circuit.random import random_circuit







if __name__=='__main__':
    provider = qk.IBMQ.load_account()
    #provider = qk.IBMQ.get_provider(
    #    hub='ibm-q-community',
    #    group='ibmquantumawards',
    #    project='open-science-22'
    #)
    provider = qk.IBMQ.get_provider(
        hub='ibm-q',
        group='open',
        project='main',
    )
    nairobi_backend = provider.get_backend('ibm_nairobi')
    oslo_backend = provider.get_backend('ibm_oslo')

    # set up qiskit simulators
    sim_nairobi_noiseless = qk.providers.aer.QasmSimulator()
    sim_nairobi_noisy = qk.providers.aer.QasmSimulator.from_backend(
        nairobi_backend
    )
    sim_oslo_noisy = qk.providers.aer.QasmSimulator.from_backend(
        oslo_backend
    )
    # set up variables

    ket_zero = opflow.Zero
    ket_one = opflow.One
    final_state_target = ket_one^ket_one^ket_zero
    target_state_matrix = final_state_target.to_matrix()
    number_of_circuits = 10
    max_depth = 200
    basis_gates=['id', 'rz', 'sx', 'x', 'cx', 'reset']

    backends = [
        sim_nairobi_noisy,
        sim_oslo_noisy
    ]

    for backend in backends:
        fidelities = np.zeros(shape=(number_of_circuits, max_depth))
        for i in range(number_of_circuits):
            print(i)
            for j in range(max_depth):
                print(j)
                circuit = random_circuit(num_qubits=3, depth=j)
                circuit = qk.compiler.transpile(
                    circuit,
                    basis_gates=basis_gates,
                    optimization_level=3,
                )
                tomography_circuits = tomo.state_tomography_circuits(
                    circuit,
                    [0, 1, 2],
                )
                job = qk.execute(
                    tomography_circuits,
                    backend,
                    shots=8192,
                )
                result = job.result()
                tomography_fitter = tomo.StateTomographyFitter(
                    result,
                    tomography_circuits,
                )
                rho_fit = tomography_fitter.fit(
                    method='lstsq',
                )
                fidelity = qk.quantum_info.state_fidelity(
                    rho_fit,
                    target_state_matrix,
                )
                #print(fidelity)
                fidelities[i, j] = fidelity

        print(backend)
        np.save(f'../data/final_runs/randomized_circuits_fidelities_max_depth{max_depth}_n_circs{number_of_circuits}_{backend}', fidelities)
