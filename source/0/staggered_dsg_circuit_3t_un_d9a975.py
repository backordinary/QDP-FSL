# https://github.com/BStar14/quantum-walk-simulation/blob/909da5a9c039cb7cd49f616c2d0c73890e146fa9/staggered_dsg_circuit_3t_un.py
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.extensions.unitary import UnitaryGate
from qiskit.quantum_info.operators import Operator


def staggered_dsg_circuit_3t_un(
    size_degree: int = 2,
    t: float = 1.,
    layers: int = 1,
    decompose: bool = False,
    linear_operation: bool = False,
    model: str = 'Heisenberg'
) -> QuantumCircuit:
    """
    Given size, time t, and the number of steps n, constructs the corresponding staggered quantum walk circuit on DSG.
    Still fixed size_degree at 2
    3 parts of tessellation
    Args:
        size_degree: The degree of DSG (size = 3**size_degree)
        t: Total simulation time
        layers: Number of layers
    Returns:
        QuantumCircuit
    """
    
    size = 3**size_degree
    node_list = np.arange(size)
    theta = np.pi*t/layers
    zeta = complex(np.cos(-2*theta), np.sin(-2*theta))
    
    if model == 'XY':
        interaction = UnitaryGate(Operator([
            [1, 0, 0, 0],
            [0, np.cos(theta), complex(0, -np.sin(theta)), 0],
            [0, complex(0, -np.sin(theta)), np.cos(theta), 0],
            [0, 0, 0, 1]])
        )
    elif model == 'Heisenberg':
        interaction = UnitaryGate(Operator([
            [1, 0, 0, 0],
            [0, (zeta+1)/2, (-1*zeta+1)/2, 0],
            [0, (-1*zeta+1)/2, (zeta+1)/2, 0],
            [0, 0, 0, 1]])
        )
    else:
        raise NotImplementedError()

    def interaction_2(size, interaction, qubits, name=None, decompose=False):
        if name!=None: interaction.name = name
        qc = QuantumCircuit(size)
        qc.append(interaction, qubits)
        if decompose: qc = transpile(qc, basis_gates=['u', 'cx'], optimization_level=2)
        return qc

    simulation = QuantumCircuit(size, size)

    simulation.x(0)

    for i in range(layers):
        
        simulation.barrier()
        
        # Staggered Quantum Walks tessellation 1
        for j in node_list[::3]:
            a = int(j+(j//3)%3)
            b = int(j+(j//3+1)%3)
            simulation.compose(
                interaction_2(size, interaction, [node_list[a], node_list[b]], 'Tessel-1', decompose),
                range(size),
                inplace=True
                )
            if linear_operation: simulation.barrier()
        for j in node_list[7::9]:
            simulation.compose(
                interaction_2(size, interaction, [j, node_list[j-5]], 'Tessel-1', decompose),
                range(size),
                inplace=True
                )
            if linear_operation: simulation.barrier()
        
        # Staggered Quantum Walks tessellation 2
        for j in node_list[::3]:
            a = int(j+(j//3+2)%3)
            b = int(j+(j//3)%3)
            simulation.compose(
                interaction_2(size, interaction, [node_list[a], node_list[b]], 'Tessel-2', decompose),
                range(size),
                inplace=True
                )
            if linear_operation: simulation.barrier()
        for j in node_list[1::9]:
            simulation.compose(
                interaction_2(size, interaction, [j, node_list[j-5]], 'Tessel-2', decompose),
                range(size),
                inplace=True
                )
            if linear_operation: simulation.barrier()
        
        # Staggered Quantum Walks tessellation 3
        for j in node_list[::3]:
            a = int(j+(j//3+1)%3)
            b = int(j+(j//3+2)%3)
            simulation.compose(
                interaction_2(size, interaction, [node_list[a], node_list[b]], 'Tessel-3', decompose),
                range(size),
                inplace=True
                )
            if linear_operation: simulation.barrier()
        for j in node_list[4::9]:
            simulation.compose(
                interaction_2(size, interaction, [j, node_list[j-5]], 'Tessel-3', decompose),
                range(size),
                inplace=True
                )
            if linear_operation: simulation.barrier()

    if not linear_operation: simulation.barrier()
    
    for i in node_list:
        simulation.measure(i, i)
       
    return simulation
