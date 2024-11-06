# https://github.com/vietzd/qc-cloud-challenges/blob/b239840b9924393f5c5e5939cea291b80e8198c5/scenario3-clustering/docker/clustering/quantumBackendFactory.py
"""
Author: Daniel Fink
Email: daniel-fink@outlook.com
"""

from qiskit import *


class QuantumBackendFactory:
    """
    A factory to create quantum backends.
    """

    @staticmethod
    def create_backend(backend_name, token=''):
        """
        Creates a quantum backend given the backend name.
        We allow two provider: aer, ibmq.
        We expect the backend_name to follow the format
        provider_instance
        i.e. ibmq_santiago, ibmq_16_melbourne, aer_qasm_simulator.
        """

        backend_name = backend_name.lower()
        if 'aer' in backend_name:
            provider = 'aer'
            instance = backend_name[4:]
        elif 'ibmq' in backend_name:
            provider = 'ibmq'
            instance = backend_name
        else:
            raise Exception('Unknown backend name specified.')

        if 'ibmq' in provider:
            if token == '':
                raise Exception('A token is needed when using ibm backends.')
            else:
                try:
                    provider = IBMQ.enable_account(token)
                    print('IBMQ_TOKEN activated')
                except:
                    IBMQ.disable_account()
                    provider = IBMQ.enable_account(token)
                    print('Disabled old and activated new IBMQ_TOKEN')
                return provider.get_backend(instance)

        elif 'aer' in provider:
            if 'qasm' in instance:
                return Aer.get_backend('qasm_simulator')
            elif 'vector' in instance:
                return Aer.get_backend('statevector_simulator')
            else:
                raise Exception('Backend provider '
                                + provider
                                + ' does not have an instance called '
                                + instance
                                + '.')
        else:
            raise Exception('Unknown backend provider.')
