# https://github.com/ampolloreno/qaoa/blob/519fff304142e5b9bc1d34e8dc1c4b2e8ac5ea37/classical_optimization/classical_optimization/terra/produce_landscape.py
"""
python produce_landscape.py filename discretization
"""
from classical_optimization.qaoa_circuits import produce_gammas_betas, maxcut_qaoa_circuit
from classical_optimization.terra.utils import write_graph, read_graph, weights, density_cost, cost
from coldquanta.qiskit_tools.modeling.neutral_atom_noise_model import create_noise_model
import numpy as np
from qiskit import Aer, execute
# I may have modified my qiskit - this adds on an attribute when I import.
from qiskit.providers.aer.extensions import snapshot_density_matrix
from recirq.qaoa.simulation import exact_qaoa_values_on_grid
import sys
import time

min_gamma = -np.pi
max_gamma = np.pi
min_beta = -np.pi/4
max_beta = np.pi/4
discretization = int(sys.argv[2])
discretization = 40
gammas, betas = produce_gammas_betas(discretization, max_gamma, max_beta, min_gamma, min_beta)
noisy = True

filename = sys.argv[1]
landscape_string = f"landscape_d{discretization}_b{max_beta}_g{max_gamma}_b{min_beta}_g{min_gamma}"
if True:
    graph = read_graph(filename)['graph']
    num_qubits = len(graph.nodes)
    start = time.time()
    if noisy:
        simulator = Aer.get_backend('qasm_simulator')
        noise_model = create_noise_model(cz_fidelity=.9)
        experiments = []
        for beta in betas:
            for gamma in gammas:
                print(beta, gamma)
                circuit = maxcut_qaoa_circuit(gammas=[gamma], betas=[beta], p=1, num_qubits=num_qubits, weights=weights(graph), measure=False, density_matrix=True)
                experiments.append(circuit)
        job = execute(experiments, backend=simulator, noise_model=noise_model)
        outputs = [result.data.snapshots.density_matrix['output'][0]['value'] for result in job.result().results]
        #The diagonal is real, so we take the first element.
        expectations = [density_cost(np.array(output)[:, :, 0], num_qubits=num_qubits, weights=weights(graph)) for output in
                        outputs]

        # expectations = [np.real(cost(job.result().get_statevector(experiment), num_qubits=num_qubits, weights=weights(graph))) for experiment in experiments]

        landscape = np.zeros((discretization, 2*discretization))
        for i, beta in enumerate(betas):
            for j, gamma in enumerate(gammas):
                landscape[i][j] = expectations[i*len(gammas) + j]
    else:
        landscape = exact_qaoa_values_on_grid(graph, num_processors=int(sys.argv[3]), xlim=(min_gamma, max_gamma), ylim=(min_beta, max_beta),
                                              x_grid_num=2 * discretization, y_grid_num=discretization)
    stop = time.time()
    write_graph(graph, {landscape_string: landscape, landscape_string + '_time': stop-start}, noisy=noisy)
else:
    print("Already computed ths one!")
