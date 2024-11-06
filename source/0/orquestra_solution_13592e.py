# https://github.com/zapatacomputing/workshop-bell-state/blob/b1220548cb5f877d0368efe346b7dd9065c740a0/steps/orquestra_solution.py
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from zquantum.core import circuits
from zquantum.core.utils import create_object
from zquantum.core.bitstring_distribution import save_bitstring_distribution
import yaml


def main(backend_specs):
    # Define registers and circuit
    q = QuantumRegister(2)
    c = ClassicalRegister(2)
    circuit = QuantumCircuit(q, c)

    # Quantum circuit starts here
    circuit.h(q[0])
    circuit.cnot(q[0], q[1])
    # Note we have to remove the Measurement to convert to Zapata's circuit format
    # End quantum circuit

    # After this point we use the power of Orquestra to use different backends!
    # Use Zapata's representation of quantum circuits
    zap_circuit = circuits.import_from_qiskit(circuit)

    # Build a backend from the specs we passed to the step
    if isinstance(backend_specs, str):
        backend_specs_dict = yaml.load(backend_specs, Loader=yaml.SafeLoader)
    else:
        backend_specs_dict = backend_specs
    backend = create_object(backend_specs_dict)

    # We can use this backend to get the bitstring distribution
    distribution = backend.get_bitstring_distribution(zap_circuit)

    # Finally, we can save the output!
    save_bitstring_distribution(distribution, "output-distribution.json")
