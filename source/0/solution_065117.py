# https://github.com/zapatacomputing/workshop-basic-vqe/blob/fac95ef33709ff8c3f408a05c6566c0f0263e999/steps/solution.py
from typing import Dict
import json
import yaml
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from zquantum.core.wip.circuits import import_from_qiskit
from zquantum.core.utils import create_object
from zquantum.core.interfaces.backend import QuantumBackend


# build_ansatz creates the ansatz for our VQE
def build_ansatz(param: Parameter) -> QuantumCircuit:
    ansatz = QuantumCircuit(1, 1)
    ansatz.ry(param, 0)
    return ansatz


# build_circuits creates the circuits required to convert to the correct computational basis
def build_circuits() -> Dict[str, QuantumCircuit]:
    # The Z circuit is already in the right computational basis
    zcircuit = QuantumCircuit(1, 1)

    # The X circuit needs to be rotated to be in the right basis
    xcircuit = QuantumCircuit(1, 1)
    xcircuit.h(0)

    # Same with the Y circuit
    ycircuit = QuantumCircuit(1, 1)
    ycircuit.u(np.pi / 2, 0, np.pi / 2, 0)

    return {
        "x": xcircuit,
        "y": ycircuit,
        "z": zcircuit,
    }


# vqe is the entrypoint for our workflow
# backend_specs is the specifications to the Orquestra backend we want to use
# coefficients is the coefficients of the Hamiltonian we are interested in
# min_value: value to start our search
# max_value: value to end our search
def vqe(backend_specs, coefficients, min_value=0, max_value=2 * np.pi):
    # Build a backend from the specs we passed to the step
    if isinstance(backend_specs, str):
        backend_specs_dict = yaml.load(backend_specs, Loader=yaml.SafeLoader)
    else:
        backend_specs_dict = backend_specs
    backend = create_object(backend_specs_dict)

    # Build the coeff we passed to the step
    if isinstance(coefficients, str):
        coefficients_dict = yaml.load(coefficients, Loader=yaml.SafeLoader)
    else:
        coefficients_dict = coefficients

    # Build the circuits
    theta = Parameter("θ")
    ansatz = build_ansatz(theta)
    circuits = build_circuits()

    # Search over our input parameters
    results, values = search(
        backend,
        ansatz,
        theta,
        circuits,
        coefficients_dict,
        min_value=min_value,
        max_value=max_value,
    )

    # Find the index of the minimum energy
    # Finding the index helps us find the parameter too.
    minimum_idx = np.argmin(results)

    data = {
        "minimum": {
            "value": results[minimum_idx],
            "theta": values[minimum_idx],
        },
        "results": results,
        "values": values.tolist(),
    }

    # Write out the results so they are in our workflow_results.json
    with open("results.json", "w") as f:
        json.dump(data, f)


# In this example, we just search over a linear space of parameters
# backend: is an Orquestra backend
# ansatz: A parameterized Qiskit circuit
# param: the parameter used in the ansatz
# circuits: a dictionary of circuits for X, Y, Z
# coefficients: a dictionary of coefficients in the Hamiltonian
# min_value: value to start our search
# max_value: value to end our search
# samples: number of samples from the quantum backend
def search(
    backend: QuantumBackend,
    ansatz: QuantumCircuit,
    param: Parameter,
    circuits: Dict[str, QuantumCircuit],
    coefficients: Dict[str, int],
    min_value=0,
    max_value=1,
    samples=10000,
):
    # Create our search space
    values = np.linspace(min_value, max_value, 100)

    results = []

    # Loop over all points in out search space
    # In real VQE workloads, you want to use an optimizer here
    for v in values:
        energy = 0
        # Loop over each coefficient
        for k, coef in coefficients.items():
            # Skip if the coefficient is 0
            if coef == 0:
                continue

            # Only run a circuit if we're not considering the "I" part of the Hamiltonian
            if k != "i":
                energy += estimate_energy(
                    backend, coef, ansatz, param, v, circuits[k], samples
                )
            else:
                energy += coef
        # Keep track of this total energy
        results.append(energy)
    # Return all the calculated energies and parameter values
    return results, values


def estimate_energy(backend, coef, ansatz, param, value, measure_circuit, samples):
    # Combine the Ansatz and measurement circuits
    circuit = ansatz + measure_circuit
    # Replace the Parameter with a value
    circuit = circuit.bind_parameters({param: value})
    # Take the expectation value, multiple by the coefficient, and to our energy value
    return coef * expectation_from_circuit(backend, circuit, samples)


# The (crudely) estimates the expectation value for a 1 qubit circuit
def expectation_from_circuit(
    backend: QuantumBackend, circuit: QuantumCircuit, samples: int
):
    # Convert the qiskit circuit
    zap_circuit = import_from_qiskit(circuit)

    # Execute the circuit using an Orquestra backend
    measurement = backend.run_circuit_and_measure(zap_circuit, n_samples=samples)
    counts = measurement.get_counts()

    # Estimate the expectation value based on the counts
    expectation_value = 0
    for bit, count in counts.items():
        sign = +1
        if bit == "1":
            sign = -1
        expectation_value += sign * count / samples
    return expectation_value
