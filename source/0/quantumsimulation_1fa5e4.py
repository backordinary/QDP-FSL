# https://github.com/Oelfol/Dynamics/blob/ed8b55297aa488444e7cd12e1352e4f1b88b525f/HeisenbergCodes/QuantumSimulation.py
###########################################################################
# QuantumSimulation.py
# Part of HeisenbergCodes
# Updated January '21
#
# Qiskit functions for creating initial state and computing expectation
# values.
###########################################################################

from qiskit import QuantumCircuit
import numpy as np
import ClassicalSimulation as cs
import HelpingFunctions as hf
import PlottingFunctions as pf


class QuantumSim:

    def __init__(self, j=0.0, bg=0.0, a=1.0, n=0, open=True, trns=False, ising=False, eps=0,
                 dev_params=[], RMfile='', unity=False):

        ###################################################################################################
        # Params:
        # (j, coupling constant); (bg, magnetic field); (a, anisotropy jz/j);
        # (n, number of sites); (open, whether open-ended chain); (states, number of basis states)
        # (unity, whether h-bar/2 == 1 (h-bar == 1 elsewise)); (ising, for ising model);
        # (trns; transverse ising); (eps, precision for trotter steps); (dev_params, for running circuit)
        # (RMfile, the filename for RM data)
        ###################################################################################################

        self.j = j
        self.bg = bg
        self.n = n
        self.states = 2 ** n
        self.a = a
        self.open = open
        self.trns = trns
        self.ising = ising
        self.unity = unity
        self.eps = eps
        self.RMfile = RMfile

        classical_h = cs.ClassicalSpinChain(j=self.j, bg=self.bg, a=self.a, n=self.n, open=self.open, unity=self.unity)
        self.h_commutes = classical_h.test_commuting_matrices()
        self.pairs_nn, autos = hf.gen_pairs(self.n, False, self.open)
        self.total_pairs = self.pairs_nn

        # For use with ibmq devices and noise models:
        self.device_params = dev_params

        # Address needed spin constants for particular paper:
        self.spin_constant = 1
        if not self.unity:
            self.spin_constant = 2

        # Params for trotter
        self.params = [self.h_commutes, self.j, self.eps, self.spin_constant, self.n, self.bg, self.trns,
                       self.total_pairs, self.ising, self.a, self.open]
    #####################################################################################

    def init_state(self, qc, ancilla, psi0):
        # Initialize a circuit in the desired spin state. Add a qubit if there is an ancilla measurement.
        state_temp, anc, index = np.binary_repr(psi0).zfill(self.n)[::-1], int(ancilla), 0
        for x in state_temp:
            if x == '1':
                qc.x(index + anc)
            index += 1
    #####################################################################################

    def magnetization_per_site_q(self, t, dt, site, psi0, trotter_alg, hadamard=False):
        qc_id = QuantumCircuit(self.n + 1, 1)
        self.init_state(qc_id, True, psi0)
        qc_id.h(0)
        if hadamard:
            qc_id.h(2)  # -------------------------------------------------------------> tachinno fig 5a

        trotter_alg(qc_id, dt, t, 1, self.params)
        hf.choose_control_gate('z', qc_id, 0, site + 1)
        qc_id.h(0)
        qc_copy = qc_id.copy()
        measurement_id = hf.run_circuit(1, qc_id, False, self.device_params, self.n, site, self.RMfile)
        measurement_noise = hf.run_circuit(1, qc_copy, True, self.device_params, self.n, site, self.RMfile)

        return measurement_id / self.spin_constant, measurement_noise / self.spin_constant
    #####################################################################################

    def all_site_magnetization_q(self, trotter_alg, total_time=0, dt=0.0, psi0=0, hadamard=False):
        data_id = hf.gen_m(self.n, total_time)
        data_noise = hf.gen_m(self.n, total_time)
        for t in range(total_time):
            for site in range(self.n):
                m_id, m_noise = self.magnetization_per_site_q(t, dt, site, psi0, trotter_alg, hadamard=hadamard)
                data_id[site, t] += m_id
                data_noise[site, t] += m_noise

        return [data_id, data_noise]
    #####################################################################################

    def total_magnetization_q(self, trotter_alg, total_time=0, dt=0.0, psi0=0):
        data_id = hf.gen_m(1, total_time)
        data_noise = hf.gen_m(1, total_time)
        data_gates = hf.gen_m(1, total_time)

        for t in range(total_time):
            total_magnetization_id = 0
            total_magnetization_noise = 0
            num_gates_total = 0
            for site in range(self.n):
                measurement_id, measurement_noise = self.magnetization_per_site_q(t, dt, site, psi0, trotter_alg)
                total_magnetization_id += measurement_id
                total_magnetization_noise += measurement_noise

            data_id[0, t] += total_magnetization_id
            data_noise[0, t] += total_magnetization_noise
            data_gates[0, t] += num_gates_total / self.n

        return [data_id, data_noise]
    #####################################################################################

    def twoPtCorrelationsQ(self, trotter_alg, total_t, dt, alpha, beta, chosen_pairs, psi0=0):

        data_real_id, data_imag_id = hf.gen_m(len(chosen_pairs), total_t), hf.gen_m(len(chosen_pairs), total_t)
        data_real_noise, data_imag_noise = hf.gen_m(len(chosen_pairs), total_t), hf.gen_m(len(chosen_pairs), total_t)
        sc = self.spin_constant ** 2

        # Remember Ideal measurement is not done for DSF because of run time
        pairs_countdown = len(chosen_pairs)
        for pair in chosen_pairs:
            pair_data_real_id = []
            pair_data_imag_id = []
            pair_data_real_noise = []
            pair_data_imag_noise = []
            pairs_countdown -= 1
            for t in range(total_t):
                print("Current Time: ", t, "Time left: ", total_t - t)
                print("Current Pair: ", pair, "Pairs left: ", pairs_countdown)
                for j in range(2):
                    qc = QuantumCircuit(self.n + 1, 1)
                    self.init_state(qc, 1, psi0)
                    qc.h(0)
                    hf.choose_control_gate(beta, qc, 0, pair[1] + 1)
                    trotter_alg(qc, dt, t, 1, self.params)
                    hf.choose_control_gate(alpha, qc, 0, pair[0] + 1)
                    hf.real_or_imag_measurement(qc, j)
                    qc_copy = qc.copy() # have to put this everywhere
                    #measurement_id = hf.run_circuit(1, qc, False, self.device_params, self.n, None, self.RMfile) / sc
                    measurement_noise = hf.run_circuit(1, qc_copy, True, self.device_params, self.n, None, self.RMfile) / sc

                    if j == 0:
                        #data_real_id[chosen_pairs.index(pair), t] += measurement_id
                        data_real_noise[chosen_pairs.index(pair), t] += measurement_noise
                        #pair_data_real_id.append(measurement_id)
                        pair_data_real_noise.append(measurement_noise)
                    elif j == 1:
                        #data_imag_id[chosen_pairs.index(pair), t] += measurement_id
                        data_imag_noise[chosen_pairs.index(pair), t] += measurement_noise
                        #pair_data_imag_id.append(measurement_id)
                        pair_data_imag_noise.append(measurement_noise)

            # Write a bunch of these:
            hf.write_data(pair_data_real_id, "data_real_id" + str(abs(pair[0] - pair[1])) + ".csv")
            hf.write_data(pair_data_real_noise, "data_real_noise" + str(abs(pair[0] - pair[1])) + ".csv")
            hf.write_data(pair_data_imag_id, "data_imag_id" + str(abs(pair[0] - pair[1])) + ".csv")
            hf.write_data(pair_data_imag_noise, "data_imag_noise" + str(abs(pair[0] - pair[1])) + ".csv")

        data_real = [data_real_id, data_real_noise]
        data_imag = [data_imag_id, data_imag_noise]
        return data_real, data_imag
    #####################################################################################

    def occupation_probabilities_q(self, trotter_alg, total_time=0, dt=0.0, psi0=0, chosen_states=[]):
        data_id = hf.gen_m(len(chosen_states), total_time)
        data_noise = hf.gen_m(len(chosen_states), total_time)
        for t in range(total_time):
            qc = QuantumCircuit(self.n, self.n)
            self.init_state(qc, 0, psi0)
            trotter_alg(qc, dt, t, 0)
            qc_copy = qc.copy()
            measurements_id = hf.run_circuit(0, qc, False, self.device_params, self.n, None, self.RMfile)
            measurements_noise = hf.run_circuit(0, qc_copy, True, self.device_params, self.n, None, self.RMfile)

            for x in chosen_states:
                data_noise[chosen_states.index(x), t] = measurements_noise[x]
            for x in chosen_states:
                data_id[chosen_states.index(x), t] = measurements_id[x]

        data = [data_id, data_noise]
        return data

    def dynamical_structure_factor(self, trotter_alg, total_t, dt, alpha, beta, psi0, k_range, w_range):

        k_min, k_max, w_min, w_max = k_range[0], k_range[1], w_range[0], w_range[1]
        res, pairs = 300, []
        k_, w_ = np.arange(k_min, k_max, (k_max - k_min) / res), np.arange(w_min, w_max, (w_max - w_min) / res)
        k = np.array(k_.copy().tolist() * res).reshape(res, res).astype('float64')
        w = np.array(w_.copy().tolist() * res).reshape(res, res).T.astype('float64')

        for j in range(self.n):
            for p in range(self.n):
                pairs.append((j, p))

        tpc_real, tpc_imag = self.twoPtCorrelationsQ(trotter_alg, total_t, dt, alpha, beta, pairs, psi0=psi0)

        # using only the noisy data - > exact from classical
        tpc_real = tpc_real[1].toarray().astype('float64')
        tpc_imag = tpc_imag[1].toarray().astype('float64')
        dsf = np.zeros_like(k).astype('float64')

        count = 0
        for jk in range(len(pairs)):
            pair = pairs[jk]
            j = pair[0] - pair[1]
            print("the code is running!!")
            theta_one = -1 * k * j
            time_sum = (np.zeros_like(w) / self.n).astype('float64')
            for t in range(total_t):
                tpc_r = tpc_real[count, t]
                tpc_i = tpc_imag[count, t]
                theta_two = w * t * dt
                theta = theta_one + theta_two
                time_sum += (np.cos(theta) * tpc_r * dt + np.sin(theta) * tpc_i * dt).astype('float64')
            count += 1
            dsf = dsf + time_sum

        # Plot noisy data
        dsf_mod = np.multiply(np.conj(dsf), dsf)
        pf.dyn_structure_factor_plotter(dsf_mod, w, k, True, self.j, k_range, res)

    def pauli_circuits(self):
        # return circuit objects representing terms in H

        # constants [x, y, z, magnetic]
        sc = self.spin_constant
        constants = [self.j / (sc**2), self.j / (sc**2), self.j * self.a / (sc**2), self.bg / sc]

        # gather pauli circuits to match with coefficients
        x_circuits, y_circuits, z_circuits = [], [], []
        for pair in self.total_pairs:
            x_circuits.append(hf.heis_pauli_circuit(pair[0], pair[1], self.n, 'x'))
            y_circuits.append(hf.heis_pauli_circuit(pair[0], pair[1], self.n, 'y'))
            z_circuits.append(hf.heis_pauli_circuit(pair[0], pair[1], self.n, 'z'))

        mag_circuits = []
        if self.bg != 0:
            for i in range(self.n):
                mag_circuits.append(hf.heis_pauli_circuit(i, 0, self.n, 'z*'))
        circs = [x_circuits, y_circuits, z_circuits, mag_circuits]

        return circs, constants


