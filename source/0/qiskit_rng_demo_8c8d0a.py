# https://github.com/alessandro-aglietti/quantum-exit/blob/dd49cd811b094657b14cf4348d3cec736ea5a065/qiskit-client/qiskit-rng-demo.py
# https://qiskit.org/documentation/apidoc/ibmq_provider.html

import logging
import os
from qiskit import IBMQ
from qiskit_rng import Generator

IBM_QUANTUM_EXPERIENCE_TOKEN = os.environ["IBM_QUANTUM_EXPERIENCE_TOKEN"]

provider = IBMQ.enable_account(IBM_QUANTUM_EXPERIENCE_TOKEN)

backends = provider.backends()

print(backends)

# TODO QRNG
# qiskit/providers/ibmq/random/ibmqrandomservice.py
# https://github.com/Qiskit/qiskit-ibmq-provider/blob/06e557b6245b1baf6865aee2a7dea5db6f301317/qiskit/providers/ibmq/random/ibmqrandomservice.py
# api/clients/random.py
# https://github.com/Qiskit/qiskit-ibmq-provider/blob/06e557b6245b1baf6865aee2a7dea5db6f301317/qiskit/providers/ibmq/api/clients/random.py
# https://github.com/qiskit-community/qiskit_rng
# notes/0.9/random_service
# https://github.com/Qiskit/qiskit-ibmq-provider/blob/1416d6de30dbf67ab9793b05a2593cc79478ccdf/releasenotes/notes/0.9/random_service-0173249bd082affd.yaml
# qiskit_rng/generator.py
# https://github.com/qiskit-community/qiskit_rng/blob/d785d0df17222c16ffb5acd5995eb697396c62fd/qiskit_rng/generator.py#L54-L76
# rng_provider = IBMQ.get_provider(hub='MY_HUB', group='MY_GROUP', project='MY_PROJECT')
# backend = backends.

# generator = Generator(backend=backend)
# output = generator.sample(num_raw_bits=1024).block_until_ready()
# print(output.mermin_correlator)
