# https://github.com/ccicconetti/qiskit-tests/blob/8e362a1d041b071b64b9630201e0bb56246c735a/utils/print_backends.py
#!/usr/bin/env python3

from qiskit import Aer
from qiskit import IBMQ

import argparse
from tutorial.utils import splitProjectInfo


def backend_configuration(backend):
    conf = backend.configuration()
    if conf:
        conf_dict = conf.to_dict()
        return (
            f"version {conf_dict['backend_version']}, n_qubits {conf_dict['n_qubits']}"
        )
    return ""


def backend_status(backend):
    status = backend.status()
    if status:
        status_dict = status.to_dict()
        operational = ""
        if not status_dict["operational"]:
            operational = "not-operational, "
        return f"{operational}{status_dict['pending_jobs']} pending jobs, status {status_dict['status_msg']}"
    return ""


parser = argparse.ArgumentParser(
    "Print IBMQ backends", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "--aer", action="store_true", default=False, help="Print the AER backends"
)
parser.add_argument(
    "--project", type=str, default="", help="Indicate a specific hub,group,project"
)
args = parser.parse_args()

if args.aer:
    print("AER backends")
    for backend in Aer.backends():
        print(f"\t{backend} ({backend_configuration(backend)})")

else:
    open_provider = IBMQ.load_account()
    if args.project == "":
        print("IBMQ backends (open)")
        for backend in open_provider.backends():
            print(
                f"\t{backend} ({backend_configuration(backend)}): {backend_status(backend)}"
            )

    else:
        (hub, group, project) = splitProjectInfo(args.project)
        print(f"IBMQ backends (hub: {hub}, group: {group}, project: {project})")
        paid_provider = IBMQ.get_provider(hub=hub, group=group, project=project)
        for backend in paid_provider.backends():
            print(
                f"\t{backend} ({backend_configuration(backend)}): {backend_status(backend)}"
            )
