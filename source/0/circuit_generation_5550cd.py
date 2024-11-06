# https://github.com/obstjn/ws-qaoa-transfer/blob/68d7085cf8ff593751ac7a13a713a73485dcf7fd/circuit_generation.py
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.algorithms import QAOA
from qiskit.opflow import I, X, Y, Z
from qiskit.compiler import transpile

import networkx as nx
from networkx import Graph


def operator_from_graph(G: Graph) -> str:
  operator = 0
  for u, v in G.edges():
    s = ['I']*len(G)  # List ['I', 'I', ..], as many Is as nodes
    s[u] = 'Z'
    s[v] = 'Z'
    s = s[::-1]  # reverse, so qbit 0 is node 0
    weight = G[u][v]['weight'] if 'weight' in G[u][v] else 1  # case for weighted graph
    operator += eval('^'.join(s)) * weight
  return -0.5 * operator


def ws_initial_state(apx_sol, eps=0.5):
  if eps < 0 or eps > 0.5:
    raise ValueError('eps must be in the range [0, 0.5].')
  if isinstance(apx_sol, str):
    # LSB of apx_sol is q0
    apx_sol = apx_sol[::-1]
  qc = QuantumCircuit(len(apx_sol))
  for i in range(len(apx_sol)):
    c = float(apx_sol[i])
    if c < eps:
      c = eps
    if c > 1-eps:
      c = 1-eps
    theta = 2 * np.arcsin(np.sqrt(c))
    qc.ry(theta, i)
  return qc


def ws_mixer(beta, apx_sol, eps=0.5):
  if eps < 0 or eps > 0.5:
    raise ValueError('eps must be in the range [0, 0.5].')
  if isinstance(apx_sol, str):
    # LSB of apx_sol is q0
    apx_sol = apx_sol[::-1]
  qc = QuantumCircuit(len(apx_sol))
  for i in range(len(apx_sol)):
    c = float(apx_sol[i])
    if c < eps:
      c = eps
    if c > 1-eps:
      c = 1-eps
    theta = 2 * np.arcsin(np.sqrt(c))
    qc.ry(-theta, i)
    qc.rz(2*beta, i)
    qc.ry(theta, i)
  return qc


def qaoa_circuit(G: Graph, apx_sol=None, eps=0.1):

  operator = operator_from_graph(G)

  gamma = Parameter(r'$\gamma$')
  beta = Parameter(r'$\beta$')

  if apx_sol is not None:
    ws_init_qc = ws_initial_state(apx_sol, eps=eps)
    ws_mixer_qc = ws_mixer(beta, apx_sol, eps=eps)
    qaoa = QAOA(reps=1, initial_state=ws_init_qc, mixer=ws_mixer_qc)
  else:
    qaoa = QAOA(reps=1)

  qaoa_qc = qaoa.construct_circuit([beta, gamma], operator)[0]

  #qaoa_qc = transpile(qaoa_qc, basis_gates=['h', 'ry', 'rz', 'rx', 'cx'])
  qaoa_qc = transpile(qaoa_qc, basis_gates=['h', 'ry', 'rzz', 'rx', 'rz'])

  return qaoa_qc
