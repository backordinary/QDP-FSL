# https://github.com/mar-be/master-thesis-code/blob/79c102a2b685eb1c54ace12cb6df31f32d4a51ae/ibmq_account.py
from typing import Dict, Optional
from qiskit import IBMQ
from qiskit.providers.ibmq.accountprovider import AccountProvider
import logger

log = logger.get_logger(__name__)


def get_provider(config:Optional[Dict]=None) -> Optional[AccountProvider]:
    """Authenticate against IBM Quantum Experience.
    Either use the specified token in the config or stored token.

    Args:
        ibmq_config (Optional[Dict], optional): If the IBMQ config dict is not None and it conatins a token, the method uses this token. Defaults to None.

    Returns:
        Optional[AccountProvider]: a authenticated session via a Provider object
    """ 
    provider = None   
    if not config is None:
        try:
            provider = IBMQ.enable_account(config["IBMQ"]["token"])
        except KeyError:
            pass
    if provider is None:
        provider = IBMQ.load_account()
    log.info("IBMQ authentification sucessful")
    return provider
