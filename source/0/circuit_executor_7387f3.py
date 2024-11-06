# https://github.com/mbeisel/VQA-workflow/blob/be94e7fcec741bf893a08b7b8871d9947c00e204/services/execution-service/app/circuit_executor.py
# ******************************************************************************
#  Copyright (c) 2020 University of Stuttgart
#
#  See the NOTICE file(s) distributed with this work for additional
#  information regarding copyright ownership.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ******************************************************************************

import time
import qiskit
from flask import jsonify
from qiskit import IBMQ, transpile, assemble, QuantumCircuit
from qiskit.providers import QiskitBackendNotFoundError, JobError, JobTimeoutError
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.ibmq.api.exceptions import RequestsApiError
from qiskit.providers.jobstatus import JOB_FINAL_STATES


def execute_circuit(input):
    circuit = input.get("circuit")
    provider = input.get("provider")
    qpu = input.get("qpu")
    credentials = input.get("credentials")
    shots = input.get("shots")
    noiseModel = input.get("noiseModel")

    if provider.lower() != 'ibm':
        return 'This service currently only supports the execution of quantum circuits on IBMQ qpus'

    # DEPRECATED: get qiskit circuit object from Json object containing the base64 encoded circuit
    # quantum_circuit = pickle.loads(codecs.decode(circuit.encode(), "base64"))
    try:
        circuit = QuantumCircuit.from_qasm_str(circuit)
    except:
        return 'The quantum circuit has to be provided as an OpenQASM 2.0 String'

    if noiseModel:
        noisy_qpu = get_qpu(credentials, noiseModel)
        noise_model = NoiseModel.from_backend(noisy_qpu)
        properties = noisy_qpu.properties()
        configuration = noisy_qpu.configuration()
        coupling_map = configuration.coupling_map
        basis_gates = noise_model.basis_gates
        transpiled_circuit = transpile(circuit, noisy_qpu)
        measurement_qubits = get_measurement_qubits_from_transpiled_circuit(transpiled_circuit)

        backend = AerSimulator()
        job = qiskit.execute(transpiled_circuit, backend=backend, coupling_map=coupling_map, basis_gates=basis_gates, noise_model=noise_model, shots=shots, optimization_level=0)
        result_counts = job.result().get_counts()
    else:
        if 'simulator' in qpu:
            ibm_qpu = AerSimulator()
            measurement_qubits = list(range(0, circuit.num_qubits))
            transpiled_circuit = transpile(circuit, backend=ibm_qpu)
        else:
            ibm_qpu = get_qpu(credentials, qpu)
            transpiled_circuit = transpile(circuit, backend=ibm_qpu)
            measurement_qubits = get_measurement_qubits_from_transpiled_circuit(transpiled_circuit)
        result_counts = execute(transpiled_circuit, shots, ibm_qpu)

    return jsonify({'counts': result_counts, 'meas_qubits': measurement_qubits})


def get_qpu(credentials, qpu):
    """Load account from token. Get backend."""
    try:
        try:
            IBMQ.disable_account()
        finally:
            provider = IBMQ.enable_account(**credentials)
            backend = provider.get_backend(qpu)
            return backend
    except (QiskitBackendNotFoundError, RequestsApiError):
        print('Backend could not be retrieved. Backend name or credentials are invalid. Be sure to use the schema credentials: {"token": "YOUR_TOKEN", "hub": "YOUR_HUB", "group": "YOUR GROUP", "project": "YOUR_PROJECT"). Note that "ibm-q/open/main" are assumed as default values for "hub", "group", "project".')
        return None


def execute(transpiled_circuit, shots, backend):
    """Execute the quantum circuit."""
    try:
        job = backend.run(assemble(transpiled_circuit, shots=shots))

        job_status = job.status()
        while job_status not in JOB_FINAL_STATES:
            print("The execution is still running")
            time.sleep(10)
            job_status = job.status()

        return job.result().get_counts()
    except (JobError, JobTimeoutError):
        return None

def get_measurement_qubits_from_transpiled_circuit(transpiled_circuit):
    qregs = transpiled_circuit.qregs
    layout = transpiled_circuit._layout.get_physical_bits()

    mapping = {}
    for k, v in layout.items():
        if v.register.name != 'ancilla':
            mapping[k] = v.index
            break

    measurement_qubits = list(mapping.keys())
    return measurement_qubits
