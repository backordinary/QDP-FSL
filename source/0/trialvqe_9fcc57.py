# https://github.com/Oelfol/Dynamics/blob/ed8b55297aa488444e7cd12e1352e4f1b88b525f/HeisenbergCodes/TrialVQE.py
###########################################################################
# TrialVQE.py
# Part of HeisenbergCodes
# Updated January '21
#
# Code/helper codes for testing various VQE strategies on Heisenberg model
# Based on reference : https://arxiv.org/pdf/1704.05018.pdf (hardware efficient ansatz)
###########################################################################

from qiskit import QuantumCircuit
import random
import math
import HelpingFunctions as hf

# TODO rewrite to allow other optimization techniques
# TODO implement calibration for a


class TrialVQE:

    def __init__(self, qchain, h_circuits, constants, d, num_updates, alpha, gamma, c, a, noisy):

        self.qchain = qchain
        self.h_circuits = h_circuits                                                # circuits per H term
        self.constants = constants                                                  # constants per H term
        self.d = d                                                                  # number of layers
        self.start_params = [[random.uniform(0, 1)*math.pi*2 for i in range(3)] for j in range(qchain.n*self.d)]
        self.end_shots = 80000 # TODO incorporate this in the final steps
        self.num_updates = num_updates
        self.alpha = alpha                                                          # SPSA param
        self.gamma = gamma                                                          # SPSA param
        self.c = c                                                                  # SPSA param
        self.a = a                                                                  # SPSA param
        self.c_k = [self.c / (x**gamma) for x in range(1, num_updates + 1)]
        self.a_k = [self.a / (y**alpha) for y in range(1, num_updates + 1)]
        self.noisy = noisy                                                          # boolean

    def optimize(self):

        data = []
        for update in range(self.num_updates):
            print('step=', update)
            bernoulli = [[random.choice([0, 1]) for i in range(3)] for j in range(self.qchain.n*self.d)]
            energy = self.get_energy(self.start_params)
            print("E", energy)
            energy_plus = self.get_energy(self.delta_params('+', update, bernoulli))
            energy_minus = self.get_energy(self.delta_params('-', update, bernoulli))
            gradient = self.get_gradient(bernoulli, energy_plus, energy_minus, update)
            self.update_params(gradient, update)
            data.append(energy)
        return data

    def delta_params(self, choice, update, bernoulli):
        # obtain theta+/theta-
        temp_params = []
        count = 0
        if choice == '+':
            for u in range(self.d):
                for j in range(self.qchain.n):
                    templist = []
                    for k in range(3):
                        templist.append(self.start_params[count][k] + self.c_k[update]*bernoulli[count][k])
                    temp_params.append(templist)
                    count = count + 1

        elif choice == '-':
            for u in range(self.d):
                for j in range(self.qchain.n):
                    templist = []
                    for k in range(3):
                        templist.append(self.start_params[count][k] - self.c_k[update]*bernoulli[count][k])
                    temp_params.append(templist)
                    count = count + 1

        return temp_params

    def get_energy(self, params):
        # obtain energy at step k
        energy = 0

        for i in range(len(self.h_circuits)):
            terms = self.h_circuits[i]
            constant = self.constants[i]

            for j in range(len(terms)):
                term_circ = terms[j]
                c, q = 1, self.qchain.n + 1
                qc = QuantumCircuit(q, c)
                qc.h(0)

                counter = 0
                for layer in range(self.d):
                    # Layer of Euler rotations
                    for q in range(self.qchain.n):
                        params_q = params[counter]
                        qc.u3(params_q[0], params_q[1], params_q[2], q + 1)
                        counter += 1
                    # Layer of entanglers
                    for pair in self.qchain.total_pairs:
                        qc.cx(pair[0] + 1, pair[1] + 1)

                qc.barrier()
                # Add controlled operation
                qc += term_circ

                # Add ancilla measurement (real)
                qc.h(0)
                qc.measure(0, 0)

                # Find result
                dp = self.qchain.device_params
                result = hf.run_circuit(1, qc, self.noisy, dp, self.qchain.n, None, self.qchain.RMfile)
                energy += result * constant

        return energy

    def get_gradient(self, bernoulli, energy_plus, energy_minus, update):
        ck = self.c_k[update]
        deltaE = (energy_plus - energy_minus) / (2 * ck)
        grad = [[deltaE * bernoulli[x][y] for y in range(3)] for x in range(self.qchain.n*self.d)]
        return grad

    def update_params(self, gradient, update):
        count = 0
        for k in range(self.d):
            for j in range(self.qchain.n):
                for i in range(3):
                    self.start_params[count][i] = self.start_params[count][i] - self.a_k[update]*gradient[count][i]
                count = count + 1
