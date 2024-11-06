# https://github.com/artgromov/quantum-lab/blob/df6f4ae2ef3fc902a46f256ef6fb58a486e320ba/helpers/__init__.py
IBMQ_PROVIDER = None


def get_ibmq_provider():
    global IBMQ_PROVIDER
    if IBMQ_PROVIDER is None:
        from qiskit import IBMQ
        IBMQ_PROVIDER = IBMQ.load_account()
    return IBMQ_PROVIDER
