# https://github.com/paulmatt67/quantum_packer_public/blob/b37d57d4a1682579dd4025115215d0ef0f2eb8bc/qiskit_methods/backends.py
from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy
from qiskit.providers.ibmq.accountprovider import AccountProvider
from typing import Optional
from qiskit_methods.config import API_TOKEN_DE, API_TOKEN_US, API_URL_DE


def load_providers_de():
    IBMQ.enable_account(API_TOKEN_DE, API_URL_DE)
    provider = IBMQ.get_provider(hub='fraunhofer-de')
    return provider


def load_providers_us(project: Optional[str] = None) -> AccountProvider:
    IBMQ.enable_account(API_TOKEN_US)
    if project is None:
        project = 'ticket'
        print(f'No project provided, using {project} per default.')
        print('Available providers:')
        for available_provider in IBMQ.providers():
            print(available_provider)
    provider = IBMQ.get_provider(hub='ibm-q-fraunhofer', project=project)
    return provider


def show_num_qubits_for_provider(provider: AccountProvider) -> None:
    result_dict = {}
    for backend in provider.backends():
        print(str(backend) + ' has ' + str(backend.configuration().n_qubits) + ' qubits')
        result_dict.update({str(backend): backend.configuration().n_qubits})
    return result_dict


def find_least_busy_backend(provider: AccountProvider, n_qubits: Optional[int] = 0) -> str:
    backend_list_filtered = provider.backends(filters=lambda x: x.configuration().n_qubits >= n_qubits)
    return least_busy(backend_list_filtered).name()

def find_simulator_backend(provider: AccountProvider, n_qubits: Optional[int] = None) -> str:
    if n_qubits is None:
        backend_list_filtered = provider.backends(filters=lambda x: x.configuration().simulator)
    else:
        backend_list_filtered = provider.backends(filters=lambda x: x.configuration().n_qubits >= n_qubits and
                                                                    x.configuration().simulator)
    return least_busy(backend_list_filtered).name()
