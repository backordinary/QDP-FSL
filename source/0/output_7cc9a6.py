# https://github.com/lweides/LNN_mapper/blob/5047d5e859f4499d46ae98727dea8744a7e80259/mapper/qasm/output.py
from typing import List, Tuple

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
import numpy as np

from mapper.gate.gate import Gate
from mapper.gate.type import Type
from mapper.state.mapping import Mapping
from mapper.state.state import State


def state_to_circuit(state: State, cregs: List[ClassicalRegister], initial_mapping: Mapping) -> Tuple[QuantumCircuit, Mapping, int, int]:
  """
    Resolves the state backwards, producing a qiskit QuantumCircuit
  """
  qubit_count = state.mapping._qubit_count
  swaps = 0
  q = QuantumRegister(qubit_count, "q")
  qc = QuantumCircuit(q, *cregs)
  gates = _resolve_state(state)
  gates, initial_mapping, free_swaps = _backpropagate_free_swaps(gates, qubit_count, initial_mapping)

  identity = initial_mapping._logical_to_physical[initial_mapping._physical_to_logical]
  if not np.all(np.arange(qubit_count) == identity):
    raise ValueError("Found illegal mapping")

  for gate in gates:
    if gate.type == Type.CNOT:
      qc.cnot(gate.q1, gate.q2)
    elif gate.type == Type.SWAP:
      qc.swap(gate.q1, gate.q2)
      swaps += 1
    elif gate.type == Type.X:
      qc.x(gate.q1)
    elif gate.type == Type.ROTATE_Z:
      qc.rz(gate.params[0], gate.q1)
    elif gate.type == Type.SQRT:
      qc.sx(gate.q1)
    elif gate.type == Type.MEASURE:
      qc.measure(gate.q1, gate.q2)
    elif gate.type == Type.BARRIER:
      qc.barrier(gate.q2) # is actually stored in q2
    elif gate.type == Type.FREE_SWAP:
      raise ValueError("A free swap gate has not been resolved!")
    elif gate.type == Type.CHECKPOINT:
      pass # nothing to do
    else:
      raise ValueError(f"Encountered invalid gate: {gate.type}")
    
  return qc, initial_mapping, swaps, free_swaps


def create_mapping_comment(mapping: Mapping) -> str:
  """
    Creates the comment for the initial mapping:
    // i 0, 2, 4, ...
    where q[0] is mapped to p[0], q[1] is mapped to p[2], q[3] is mapped to p[4], etc.
  """
  comment = "// i "
  mapping_text = " ".join(str(q) for q in mapping._logical_to_physical)
  return comment + mapping_text


def _backpropagate_free_swaps(gates: List[Gate], qubit_count: int, initial_mapping: Mapping) -> Tuple[List[Gate], Mapping, int]:
  """
    Applies the free swap gates, which essentially compute an initial mapping
  """
  count = 0
  for index, gate in enumerate(gates):
    if gate.type != Type.FREE_SWAP:
      continue
    
    count += 1
    
    f_swap = gate
    p1, p2 = initial_mapping.physical_to_logical(f_swap.q1, f_swap.q2)
    initial_mapping.swap_inplace(p1, p2)

    for g in gates[0:index]:

      if g.type == Type.FREE_SWAP: # no need to modify those
        continue

      if g.q1 == f_swap.q1:
        g.q1 = f_swap.q2
      elif g.q1 == f_swap.q2:
        g.q1 = f_swap.q1

      if g.type != Type.BARRIER and g.q2 == f_swap.q1 and g.type != Type.MEASURE:
        g.q2 = f_swap.q2
      elif g.type != Type.BARRIER and g.q2 == f_swap.q2 and g.type != Type.MEASURE:
        g.q2 = f_swap.q1
  
  return list(filter(lambda g: g.type != Type.FREE_SWAP, gates)), initial_mapping, count


def _resolve_state(state: State) -> List[Gate]:
  """
    Reverses the states, outputing a gate each
  """
  gates = []

  while state:
    if state.output:
      gates.append(state.output)
    state = state.parent

  # the first state has no output
  return list(reversed(gates))[1:]
  