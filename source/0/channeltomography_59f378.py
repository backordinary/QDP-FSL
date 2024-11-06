# https://github.com/dimedrolling/QuantumCode/blob/17e009819b9883fa2fb77e084166047935f58181/ChannelTomography.py
import qiskit

# Needed for functions
import time
import numpy as np

import matplotlib.pyplot as plt
from qiskit.visualization import plot_bloch_multivector

import qiskit.quantum_info as qi
from qiskit.quantum_info import partial_trace, Statevector

# Tomography functions
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit.ignis.verification.tomography import process_tomography_circuits, ProcessTomographyFitter


def channel_from_choi(choi_mat, dens_mat):
    if type(dens_mat) == list or type(dens_mat) == np.ndarray:
        dens_mat = qi.DensityMatrix(dens_mat)
    tr_dens = dens_mat.to_operator().transpose()
    mat = choi_mat.dot(tr_dens)
    chan = (partial_trace(mat.data, [1])).data
    return chan


class ChannelProc:
    def __init__(self, choi):
        self.choi = choi

    def outvec(self, in_vec):
        expected = channel_from_choi(self.choi, in_vec)
        #         print(expected.trace())
        expected = expected / expected.trace()
        #         plot_bloch_multivector(expected)
        expected = qi.DensityMatrix(expected)
        return expected


# for circuit and backend return channel ( works only for teleportation-like systems)
def channel_from_proc(circuit, backend, noise_model=None, measured=1, prepared=0, input_job=None):
    qpt_circs = process_tomography_circuits(circuit, [measured], prepared_qubits=[prepared])
    if input_job is None:
        job = qiskit.execute(qpt_circs, backend, shots=4000, noise_model=noise_model)
    else:
        job = input_job
    # Extract tomography data so that counts are indexed by measurement configuration
    qpt_tomo = ProcessTomographyFitter(job.result(), qpt_circs)
    print(qpt_tomo.data)

    # Tomographic reconstruction

    t = time.time()
    choi_fit = qpt_tomo.fit(method='lstsq')
    print('Fit time:', time.time() - t)
    print('Average gate fidelity: F = {:.5f}'.format(qi.average_gate_fidelity(choi_fit)))

    #     expected = ChannelFromChoi(choi_fit,[1/np.sqrt(2),-1/np.sqrt(2)])
    #     print(expected)
    #     plot_bloch_multivector(expected)
    chan = ChannelProc(choi_fit)
    return chan, job


def final_state_tomography(circuit, backend, noise_model=None, measured=1, prepared=0, input_job=None):
    # Generate circuits and run on simulator
    #     t = time.time()

    # Generate the state tomography circuits.
    qst_circuit = state_tomography_circuits(circuit, [measured])

    # Execute
    job = qiskit.execute(qst_circuit, backend=backend, noise_model=noise_model, shots=5000)
    #     print('Time taken:', time.time() - t)

    # Fit result
    circ_fitter_teleport = StateTomographyFitter(job.result(), qst_circuit)

    # Perform the tomography fit
    # which outputs a density matrix
    rho_fit_teleport = circ_fitter_teleport.fit(method='lstsq')
    rho = qi.DensityMatrix(rho_fit_teleport)
    plot_bloch_multivector(rho.data)
    return rho, job


def trans_sphere_from_channel(channel, in_states=None, noise_model=None, measured=4):
    # Create a sphere
    u, v = np.mgrid[0:2 * np.pi:12j, 0:np.pi:16j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)

    if in_states is None:
        xx = np.copy(x)
        yy = np.copy(y)
        zz = np.copy(z)
        for i in range(len(z)):
            for j in range(len(z[0])):
                in_state = qi.Pauli('I').to_matrix() / 2 + qi.Pauli('X').to_matrix() * x[i, j] / 2 + qi.Pauli(
                    'Y').to_matrix() * y[i, j] / 2 + qi.Pauli('Z').to_matrix() * z[i, j] / 2
                sub_rho = channel.outvec(in_state)
                #                 sub_rho = qi.DensityMatrix(in_state)
                xx[i, j] = (qi.Pauli('X').to_matrix().dot(sub_rho)).trace().real
                yy[i, j] = (qi.Pauli('Y').to_matrix().dot(sub_rho)).trace().real
                zz[i, j] = (qi.Pauli('Z').to_matrix().dot(sub_rho)).trace().real
    else:
        xx, yy, zz = np.array([]), np.array([]), np.array([])
        in_x, in_y, in_z = [], [], []
        for in_state in in_states:
            sub_rho = channel.outvec(in_state)
            #             sub_rho = qi.DensityMatrix(in_state)
            xx = np.append(xx, (qi.Pauli('X').to_matrix().dot(sub_rho)).trace().real)
            yy = np.append(yy, (qi.Pauli('Y').to_matrix().dot(sub_rho)).trace().real)
            zz = np.append(zz, (qi.Pauli('Z').to_matrix().dot(sub_rho)).trace().real)

            in_x, in_y, in_z = np.append(in_x,
                                         (qi.Pauli('X').to_matrix().dot(qi.DensityMatrix(in_state))).trace().real), \
                               np.append(in_y,
                                         (qi.Pauli('Y').to_matrix().dot(qi.DensityMatrix(in_state))).trace().real), \
                               np.append(in_z, (qi.Pauli('Z').to_matrix().dot(qi.DensityMatrix(in_state))).trace().real)

    #    Set colours and render
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #     ax.plot_wireframe(xx, yy, zz, color="k", rstride=1, cstride=2,linewidth=1)
    ax.scatter(xx, yy, zz, color="k", s=1)
    #     ax.plot( xx, yy, zz, color='c', alpha=0.3, linewidth=1)
    if in_states is not None:
        ax.scatter(in_x, in_y, in_z, color="b", s=1)
    else:
        ax.plot_wireframe(x, y, z, color="r", rstride=1, cstride=2, linewidth=1)
        ax.plot_wireframe(xx, yy, zz, color="k", rstride=1, cstride=2, linewidth=1)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    # ax.set_aspect("equal")
    plt.tight_layout()

    plt.show()


def prep_states(theta, phi):
    phi_arr = np.linspace(-np.pi, np.pi, num=theta)
    a_arr = np.linspace(0, np.pi, num=phi)
    test_states = []
    for a_i in a_arr:
        for phi_i in np.exp(1j * phi_arr):
            test_states.append([phi_i * np.cos(a_i), np.sin(a_i)])
    return test_states


def counts_to_state_vec(counts):
    state = [np.sqrt(counts['0'] * 1), np.sqrt(counts['1'] * 1)]

    state = state / np.sqrt(counts['0'] + counts['1'])
    state = state.tolist()
    return state


def coord_from_matrix(state):
    x = (qi.Pauli('X').to_matrix().dot(state)).trace().real
    y = (qi.Pauli('Y').to_matrix().dot(state)).trace().real
    z = (qi.Pauli('Z').to_matrix().dot(state)).trace().real
    return [x, y, z]
