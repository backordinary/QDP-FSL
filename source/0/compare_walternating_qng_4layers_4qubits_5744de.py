# https://github.com/vutuanhai237/UC-VQA/blob/a913edd393a291b54afa9b7ae9dfed71157bbdcb/codes/tomography/multiprocessing_scripts/compare_Walternating_qng_4layers_4qubits.py
import qiskit
import numpy as np
import sys
sys.path.insert(1, '../')
import qtm.base, qtm.constant, qtm.ansatz, qtm.fubini_study, qtm.encoding
import multiprocessing


def run_walternating(num_layers, num_qubits):

    thetas = np.ones(int(num_qubits*num_layers/2) + 3 * num_layers * num_qubits)
    psi = 2 * np.random.uniform(0, 2*np.pi, (2**num_qubits))
    psi = psi / np.linalg.norm(psi)
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    qc.initialize(psi, range(0, num_qubits))

    loss_values = []
    thetass = []
    for i in range(0, 400):
        if i % 20 == 0:
            print('W_alternating: ', i)
        
        G = qtm.fubini_study.qng(qc.copy(), thetas, qtm.ansatz.create_Walternating_layerd_state, num_layers)
        grad_loss = qtm.base.grad_loss(
            qc, 
            qtm.ansatz.create_Walternating_layerd_state,
            thetas, num_layers = num_layers)
        thetas = np.real(thetas - qtm.constant.learning_rate*(np.linalg.pinv(G) @ grad_loss)) 
        thetass.append(thetas.copy())
        qc_copy = qtm.ansatz.create_Walternating_layerd_state(qc.copy(), thetas, num_layers)  
        loss = qtm.loss.loss_basis(qtm.base.measure(qc_copy, list(range(qc_copy.num_qubits))))
        loss_values.append(loss)


    traces = []
    fidelities = []

    for thetas in thetass:
        qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
        qc = qtm.ansatz.create_Walternating_layerd_state(
            qc, thetas, num_layers=num_layers).inverse()
        psi_hat = qiskit.quantum_info.Statevector.from_instruction(qc)
        # Calculate the metrics
        trace, fidelity = qtm.base.get_metrics(psi, psi_hat)
        traces.append(trace)
        fidelities.append(fidelity)
    print('Writting ... ' + str(num_layers) + ' layers,' + str(num_qubits) +
          ' qubits')

    np.savetxt("../../experiments/tomography/tomography_walternating_" + str(num_layers) + "/" +
               str(num_qubits) + "/loss_values_qng.csv",
               loss_values,
               delimiter=",")
    np.savetxt("../../experiments/tomography/tomography_walternating_" + str(num_layers) + "/" +
               str(num_qubits) + "/thetass_qng.csv",
               thetass,
               delimiter=",")
    np.savetxt("../../experiments/tomography/tomography_walternating_" + str(num_layers) + "/" +
               str(num_qubits) + "/traces_qng.csv",
               traces,
               delimiter=",")
    np.savetxt("../../experiments/tomography/tomography_walternating_" + str(num_layers) + "/" +
               str(num_qubits) + "/fidelities_qng.csv",
               fidelities,
               delimiter=",")


if __name__ == "__main__":
    # creating thread

    num_layers = [4]
    num_qubits = [4]
    t_walternatings = []

    for i in num_layers:
        for j in num_qubits:
            t_walternatings.append(
                multiprocessing.Process(target=run_walternating, args=(i, j)))

    for t_walternating in t_walternatings:
        t_walternating.start()

    for t_walternating in t_walternatings:
        t_walternating.join()

    print("Done!")