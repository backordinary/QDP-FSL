# https://github.com/turtledude1818/Quantum-env/blob/d108761f84e76d242cd9c6c36602e296d42011ee/env.py
import gym
from gym import spaces
import numpy as np
from qiskit import QuantumCircuit, execute, transpile, Aer, IBMQ
from qiskit.visualization import *
import matplotlib.pyplot as plt
from random import randint, randrange, uniform
from math import pi

class QuantumEnv(gym.Env):
    def __init__(self, simulator=True):
        super(QuantumEnv, self).__init__()
        self.num_q = 4 #randint(1, 4)
        self.gate_dict = {'x': 0, 'y': 1, 'z': 2}
        self.gates = []
        self.predictions = []
        self.gen_circuit()
        self.output = execute(self.qc, backend=Aer.get_backend('qasm_simulator'), shots=1024).result().get_counts(self.qc)
        self.curr_comb = 0
        self.total_left = 1
        self.action_space = spaces.Box(low=0, high=1, shape=(1,)) # frequency of qubit combination
        self.observation_space = spaces.Tuple(((spaces.Tuple((spaces.Discrete(3), spaces.Box(low=0, high=2*pi,shape=(1,)), 
        spaces.Discrete(self.num_q)))) for x in range(len(self.gates)))) # list of gates
    def add_gate(self, gate):
        bit = randrange(0,self.num_q)
        theta = uniform(0, 2*pi)
        self.gates.append((gate, theta, bit))
        if gate == 0:
            self.qc.rx(theta, bit)
        elif gate == 1:
            self.qc.ry(theta, bit)
        elif gate == 2:
            self.qc.rz(theta, bit)
    def gen_circuit(self):
        self.qc = QuantumCircuit(self.num_q, self.num_q)
        for i in range(randint(2,16)):
            self.add_gate(randint(0,2))
        self.qc.measure_all()
    def step(self, action):
        self._take_action(action)
        obs = self._observe()
        done = self.curr_comb == 2**self.num_q-1
        reward = 1-abs(self.predictions[self.curr_comb] - (self.output[format(self.curr_comb, 'b').zfill(4) + " 0000"] 
        if format(self.curr_comb, 'b').zfill(4) + "0000" in self.output else 0))
        self.curr_comb += 1

        return obs,done,reward, {}
    def reset(self):
        self.gates = []
        self.predictions = []
        self.gen_circuit()
        self.output = execute(self.qc, backend=Aer.get_backend('qasm_simulator'), shots=1024).result().get_counts(self.qc)
        self.curr_comb = 0
        self.total_left = 1
        self.observation_space = spaces.Tuple(((spaces.Tuple((spaces.Discrete(3), spaces.Box(low=0, high=2*pi,shape=(1,)), 
        spaces.Discrete(self.num_q)))) for x in range(len(self.gates))))
        return self._observe()
    def _observe(self):
        return self.gates
    def _take_action(self, action):
        self.predictions.append(action[0])
        self.total_left -= action[0]
        self.action_space = spaces.Box(low=0, high=self.total_left, shape=(1,))
        