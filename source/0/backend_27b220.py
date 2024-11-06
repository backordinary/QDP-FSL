# https://github.com/hjaleta/QPE/blob/7dc9862c3cd3404d4dee00cae3a3f91eebd73216/src/QPE/Backend.py
import json
from qiskit import IBMQ, Aer
from qiskit.providers.ibmq import least_busy
from qiskit.providers.aer import AerSimulator
from qiskit.test.mock import FakeTokyo
from quantuminspire.credentials import enable_account as load_QI
from quantuminspire.qiskit import QI

def Login(service="IBMQ", token_path = "API_tokens.json", token = ""):
    """
    Logs the user into the desired backend provider

    Parameters
    ---------
        service (str, optional): 
            The service to be used for the backend. Defaults to "IBMQ". Possible other choice is 'QI' (QuantumInspire)
        token_path (str, optional): 
            Filepath to a json-file where the API tokens are stored. Defaults to "API_tokens.json".
        token (str, optional): 
            An API-token. Use if you dont have the token file. Must correspond to the chosen service. Defaults to ""

    Raises
    ------
        KeyError: 
            If using an unsupported/non-existing service
        FileNotFoundError: 
            If file with tokens can't be found
        KeyError: 
            If token file is found, but does not contain the right API_tokens
    """

    if service not in ["IBMQ", "QI"]:
        raise KeyError('Choose service "IBMQ" or "QI"')

    if token:
        API_key = token
    else:
        try:
            with open(token_path, "r") as f:
                token_dict = json.load(f)
        except:
            raise FileNotFoundError(f"No json file {token_path} found")
        try:
            API_key = token_dict[service]
        except:
            raise KeyError(f"No token in {token_path} for service {service}")

    if service == "IBMQ":
        try:
            IBMQ.load_account()
        except:
            API_key = token_dict["IBMQ"]
            IBMQ.enable_account(API_key)
    elif service == "QI":
        API_key = token_dict["QI"]
        load_QI(API_key)
        QI.set_authentication()

def get_backend(service="IBMQ", backend_name = "", flags:list = [], token_path = "API_tokens.json", token = ""):
    """A function that acquires a backend object based upon the given parameters.

    Parameters
    ---------
        service (str, optional): 
            The service which the backend  Defaults to "IBMQ".
        backend_name (str, optional): 
            The name of the desired backend. Defaults to "".
        flags (List[str], optional):
            The string are flags that one can use to filter backends. Defaults to [].
        token_path (str, optional):
            A directory where API_tokens are stored. Defaults to "API_tokens.json".
        token (str, optional):
            An API_token corresponding to the used 'service'. Defaults to "".

    Raises
    ------
        KeyError: 
            If using an unsupported/non-existing service

    Returns
    -------
        backend: 
            A backend which can be used within qiskit's framework
    """

    if service not in ["IBMQ", "QI", "local"]:
        raise KeyError('Choose service "IBMQ","QI" or "local"')

    if service in ["IBMQ", "QI"]:
        Login(service, token_path, token)
    
    sim = lambda x: x.configuration().simulator,
    Q5 = lambda x: x.configuration().n_qubits == 5 and not x.configuration().simulator

    if service == "IBMQ":
        provider = IBMQ.get_provider()
        if backend_name == "":
            if "5qubit" in flags:
                backend = least_busy(provider.backends(
                    filters= Q5
                    ))
            elif "simulator" in flags:
                backend = least_busy(provider.backends(
                    filters= sim
                    ))
        else:
            backend = provider.get_backend(backend_name)
    
    elif service == "QI":
        backend = QI.get_backend("Starmon-5")
        print(backend)
    
    elif service == "local":
        backend = Aer.get_backend("aer_simulator")

    return backend

filters = {
    "simulator": lambda x: x.configuration().simulator,
    "5qubit": lambda x: x.configuration().n_qubits == 5
                                   and not x.configuration().simulator
 }

if __name__ == "__main__":
    Login("IBMQ")
    #print(IBMQ.providers())
    backend = get_backend("IBMQ")
    