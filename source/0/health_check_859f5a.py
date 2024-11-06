# https://github.com/Jpifer13/qiskit_computing/blob/84b42fd06bdc6089c5c0c3af5b50095c1153f141/application/services/health_check.py
from typing import Optional

from qiskit import IBMQ

def healthy(imbq: IMBQ) -> Optional[str]:
    """
    This is a health check service that when called will return whether or not 
    the connection to qiskit services is working

    Args:
        imbq (IMBQ): IMBQ object
    Returns:
        str: Healthy statement
        None: no connection
    """
    pass


