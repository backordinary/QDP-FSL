# https://github.com/Akimasa11/2022_microsoft_ionq_challenge/blob/d99697b718a21362ac463c0e0078f969d546bfba/ValentinesEnv.py
from qiskit import *
import numpy as np
from qiskit.extensions import Initialize
from qiskit.quantum_info import Statevector
import gym
from gym import spaces
import random

VICTORY_INNER_PRODUCT = 0.99


simulator = Aer.get_backend('statevector_simulator')

statevectors = [
    np.array([1, 0, 0, 0]),
    np.array([0, 1, 0, 0]),
    np.array([0, 0, 1, 0]),
    np.array([0, 0, 0, 1]),
    np.array([1, 0, 0, 1])/np.sqrt(2),
    np.array([1, 1, 0, 0])/np.sqrt(2),
    np.array([1, 0, 1, 0])/np.sqrt(2),
    np.array([0, 1, 0, 1])/np.sqrt(2),
    np.array([0, 0, 1, 1])/np.sqrt(2),
    np.array([0, 1, 1, 0])/np.sqrt(2),
    np.array([-1, 0, 0, 1])/np.sqrt(2),
    np.array([-1, 1, 0, 0])/np.sqrt(2),
    np.array([-1, 0, 1, 0])/np.sqrt(2),
    np.array([0, -1, 0, 1])/np.sqrt(2),
    np.array([0, 0, 1, -1])/np.sqrt(2),
    np.array([0, -1, 1, 0])/np.sqrt(2),
    np.array([1, -1, 1, 1])/2,
    np.array([1, 1, -1, 1])/2,
    np.array([1, 1, 1, -1])/2,
    np.array([-1, 1, 1, 1])/2,
    np.array([-1, -1, 1, 1])/2,
    np.array([1, -1, -1, 1])/2,
    np.array([-1, 1, -1, 1])/2,
    np.array([1, 1, 1, 1])/2
]


def getStatevector(statevector_id):
    return Statevector(statevectors[statevector_id])


def getIdFromStatevector(statevector):
    for i in range(len(statevectors)):
        if statevector == Statevector(statevectors[i]) or statevector == Statevector(-statevectors[i]):
            return i


def initializeQuantumCircuit(statevector_ids):
    qc = QuantumCircuit(2)
    player_statevector = getStatevector(statevector_ids[0])
    init_gate = Initialize(player_statevector)
    qc.append(init_gate, [0, 1])
    return qc


gates = ["h0", "h1", "x0", "x1", "z0", "z1", "cx0", "cx1", "swap"]


def advanceTurnAux(gate, target_statevector, qc):
    if(gate == "h0"):
        qc.h(0)
    elif(gate == "h1"):
        qc.h(1)
    elif(gate == "x0"):
        qc.x(0)
    elif(gate == "x1"):
        qc.x(1)
    elif(gate == "z0"):
        qc.z(0)
    elif(gate == "z1"):
        qc.z(1)
    elif(gate == "cx0"):
        qc.cx(0, 1)
    elif(gate == "cx1"):
        qc.cx(1, 0)
    elif(gate == "swap"):
        qc.swap(0, 1)
    else:
        return Exception
    qc.barrier()
    # need to run circuit on simulator here
    job_sim = execute(qc, simulator, shots=1024)
    currStatevector = job_sim.result().get_statevector(qc)
    # compute inner product using get_statevector
    innerProd = currStatevector.inner(target_statevector)
    # return pretty version of statevector (so that player can see their current statevector) and inner product
    return [qc, currStatevector, innerProd]


def advanceTurn(gate, target_statevector_id, qc):
    target_statevector = getStatevector(target_statevector_id)
    return advanceTurnAux(gate, target_statevector, qc)


class ValentinesEnv(gym.Env):
    def __init__(self):
        super(ValentinesEnv, self).__init__()
        self.current_step = 0
        self.games = 0
        self.reward_range = (-1, 20)
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(49,), dtype=np.double)

    def step(self, gate):
        # Execute one time step within the environment
        self.current_step += 1
        self.qc, curr_statevector, new_inner_product = advanceTurn(
            gates[gate], self.target_sv_id, self.qc)

        reward = 0

        if abs(new_inner_product) > VICTORY_INNER_PRODUCT or self.current_step == 20:
            self.games += 1
            reward = 20 - self.current_step
            # if reward > 0:
            # print("\n" + str(reward))
            self.done = True

        self.prev_inner_prod = abs(new_inner_product)
        self.player_sv_id = getIdFromStatevector(curr_statevector)

        return self.get_observation(), reward, self.done, {}

    def reset(self, player_sv_id=None, target_sv_id=None):
        # Reset the state of the environment to an initial state
        self.done = False
        self.current_step = 0

        if player_sv_id is not None:
            self.player_sv_id = player_sv_id
            self.target_sv_id = target_sv_id
        else:
            self.player_sv_id = random.randint(0, 23)
            self.target_sv_id = random.randint(0, 23)
            while(self.player_sv_id == self.target_sv_id):
                self.target_sv_id = random.randint(0, 23)

        self.qc = initializeQuantumCircuit(
            [self.player_sv_id, self.target_sv_id])

        player_sv = getStatevector(self.player_sv_id)
        target_sv = getStatevector(self.target_sv_id)
        self.prev_inner_prod = player_sv.inner(target_sv)

        return self.get_observation()

    def get_observation(self):
        observation = [0] * 49
        observation[self.player_sv_id] = 1
        observation[24 + self.target_sv_id] = 1
        observation[48] = self.prev_inner_prod
        return observation
