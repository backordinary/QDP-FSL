# https://github.com/rum-yasuhiro/experiments_crosstalk_multitasking/blob/2728ed075deb4b6695233612cd4ab80f22ec511d/utils/get_backend.py
from qiskit import IBMQ


def get_IBM_backend(backend_name, reservations: bool = False):

    try:
        if reservations:
            provider = IBMQ.get_provider(
                hub="ibm-q-utokyo", group="keio-internal", project="reservations"
            )
        else:
            provider = IBMQ.get_provider(
                hub="ibm-q-utokyo", group="keio-internal", project="keio-students"
            )
    except:
        IBMQ.load_account()
        if reservations:
            provider = IBMQ.get_provider(
                hub="ibm-q-utokyo", group="keio-internal", project="reservations"
            )
        else:
            provider = IBMQ.get_provider(
                hub="ibm-q-utokyo", group="keio-internal", project="keio-students"
            )

    backend = provider.get_backend(backend_name)

    return backend
