# https://github.com/Jpifer13/qiskit_computing/blob/84b42fd06bdc6089c5c0c3af5b50095c1153f141/application/__init__.py
from qiskit import IBMQ
from .runner import Runner

def create_app():
    # Check if account is currently loaded and if it isn't create one
    if not IBMQ.active_account:
        IBMQ.save_account('MY_API_TOKEN')

    runner = Runner()

    return runner