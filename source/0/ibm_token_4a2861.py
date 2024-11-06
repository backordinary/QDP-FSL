# https://github.com/mohsenhariri/qisirq/blob/9b6e839365a87c099f85d8b97540951b5dedd63d/pkg/ibm_token.py
from os import getenv

from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy

# from qiskit_ibm_provider import IBMProvider, accounts


# def init() -> None:
#     ibm_token = getenv("IBMTOKEN")
#     if ibm_token is None:
#         raise Exception("need a token")

#     try:
#         IBMProvider.save_account(token=ibm_token)
#     except accounts.exceptions.AccountAlreadyExistsError as msg:
#         print(msg)
#     except Exception as err:
#         raise err


class Connection:
    def __init__(self) -> None:

        ibm_token = getenv("IBMTOKEN")
        if ibm_token is None:
            raise Exception("need a token")
        try:
            IBMQ.save_account(token=ibm_token)
        except Exception as err:
            raise err

    def connect(self, n_qubits):
        """_summary_

        Args:
            n_qubits (_type_): _description_

        Returns:
            _type_: _description_
        """
        print("un")
        IBMQ.load_account()
        print("List of providers: ", IBMQ.providers())

        provider = IBMQ.get_provider(hub="ibm-q")

        backend = least_busy(
            provider.backends(
                filters=lambda x: x.configuration().n_qubits >= (n_qubits + 1)
                and not x.configuration().simulator
                and x.status().operational == True  # Need an operational backend
            )
        )
        print("least busy backend: ", backend)

        return backend
