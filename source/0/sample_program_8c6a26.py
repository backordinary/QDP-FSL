# https://github.com/IEEEServices/hackathon-2021/blob/5c0bc382e11255925ffc5fdb7cb139440419fa49/runtime_local/sample_program.py
# This code is part of qiskit-runtime.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A sample runtime program that submits random circuits for user-specified iterations."""

import random

from qiskit import transpile
from qiskit.circuit.random import random_circuit


def prepare_circuits(backend):
    """Generate a random circuit.

    Args:
        backend (qiskit.providers.Backend): Backend used for transpilation.

    Returns:
        qiskit.QuantumCircuit: Generated circuit.
    """
    circuit = random_circuit(num_qubits=5, depth=4, measure=True, seed=random.randint(0, 1000))
    return transpile(circuit, backend)


def main(backend, user_messenger, **kwargs):
    """Main entry point of the program.

    Args:
        backend (qiskit.providers.Backend): Backend to submit the circuits to.
        user_messenger (qiskit.providers.ibmq.runtime.UserMessenger): Used to communicate with the
            program consumer.
        kwargs: User inputs.
    """
    iterations = kwargs.pop("iterations", 5)
    for it in range(iterations):
        qc = prepare_circuits(backend)
        result = backend.run(qc).result()
        user_messenger.publish({"iteration": it, "counts": result.get_counts()})

    user_messenger.publish("All done!", final=True)
