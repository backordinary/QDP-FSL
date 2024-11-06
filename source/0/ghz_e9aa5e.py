# https://github.com/jwoehr/ghz_local/blob/f3fcb0e116229d48845f2cae07271939756d66b4/ghz.py
"""ghz.py
Greenberger, Horne, and Zeilinger
Based on
https://quantumexperience.ng.bluemix.net/qx/tutorial?sectionId=full-user-guide&page=003-Multiple_Qubits_Gates_and_Entangled_States~2F003-GHZ_States
Copyright 2019 Jack Woehr jwoehr@softwoehr.com PO Box 51, Golden, CO 80402-0051
BSD-3 license -- See LICENSE which you should have received with this code.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES.
"""

from qiskit import execute
from qiskit.tools.monitor import job_monitor

import argparse
import sys
import datetime
import ghz_circuits as ghzc

explanation = """GHZ Local : Local Realism in Greenberger, Horne, and Zeilinger
Copyright 2019 Jack Woehr jwoehr@softwoehr.com PO Box 51, Golden, CO 80402-0051
BSD-3 license -- See LICENSE which you should have received with this code.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES.
"""

long_explan = """GHZ Local : Local Realism in Greenberger, Horne, and Zeilinger
Copyright 2019 Jack Woehr jwoehr@softwoehr.com PO Box 51, Golden, CO 80402-0051
BSD-3 license -- See LICENSE which you should have received with this code.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES.

Default is to run on genuine IBM Q quantum processor.

Type -h or --help for important options information
"""

parser = argparse.ArgumentParser(description=explanation)
group = parser.add_mutually_exclusive_group()
group.add_argument("-q", "--ibmq", action="store_true",
                   help="Use genuine IBMQ processor (default)")
group.add_argument("-s", "--sim", action="store_true",
                   help="Use IBMQ qasm simulator")
group.add_argument("-a", "--aer", action="store_true",
                   help="User QISKit aer simulator")
parser.add_argument("-b", "--backend", action="store",
                    help="""genuine qpu backend to use, default is least busy
                    of large enough devices""")
parser.add_argument("--token", action="store",
                    help="Use this token if a --url argument is also provided")
parser.add_argument("--url", action="store",
                    help="Use this url if a --token argument is also provided")
parser.add_argument("-t", "--test", action="store_true",
                    help="Only print circuits in qasm and drawing and exit.")
parser.add_argument("-u", "--usage", action="store_true",
                    help="Show long usage message and exit 0")
parser.add_argument("-v", "--verbose", action="count", default=0,
                    help="Increase verbosity each -v up to 3")
parser.add_argument("--qasm", action="store_true",
                    help="Print circuit name and qasm code for each circuit")
parser.add_argument("--draw", action="store_true",
                    help="Draw each circuit")
parser.add_argument("--to_err", action="store_true",
                    help="Redirect job monitor messages to stderr")
parser.add_argument("--quiet", action="store_true",
                    help="Suppress job monitor messages")
parser.add_argument("--results", action="store_true",
                    help="Print raw results for each experiment")

args = parser.parse_args()

if args.usage:
    print(long_explan)
    exit(0)

if (args.token and not args.url) or (args.url and not args.token):
    print('--token and --url must be used together or not at all', file=sys.stderr)
    exit(1)


def verbosity(text, count):
    """Verbose error messages by level"""
    if args.verbose >= count:
        print(text)


# If --test, just print circuits and exit
if args.test:
    ghzc.test()
    exit(0)

# Choose backend
backend = None

if args.aer:
    # Import Aer
    from qiskit import BasicAer
    # Run the quantum circuit on a statevector simulator backend
    backend = BasicAer.get_backend('statevector_simulator')
    # backend = BasicAer.get_backend('unitary_simulator')
else:
    from qiskit import IBMQ
    if args.token:
        provider = IBMQ.enable_account(args.token, url=args.url)
    else:
        provider = IBMQ.load_account()

    # Choose backend and connect
    if args.sim:
        backend = provider.get_backend('ibmq_qasm_simulator')
    else:
        if args.backend:
            backend = provider.get_backend(args.backend)
        else:
            from qiskit.providers.ibmq import least_busy
            large_enough_devices = provider.backends(filters=lambda x: x.configuration().n_qubits > 5
                                                     and not x.configuration().simulator)
            backend = least_busy(large_enough_devices)
        verbosity("The best backend is " + backend.name(), 1)

verbosity("Backend is " + backend.name(), 1)

if backend is None:
    print("No backend available, quitting.")
    exit(100)

# Prepare job
# ###########

# Number of shots to run the program (experiment); maximum is 8192 shots.
shots = 1024
# Maximum number of credits to spend on executions.
max_credits = 3

# Create circuits
circuits = [ghzc.GHZState3Q(), ghzc.GHZ_YYX(), ghzc.GHZ_YXY(),
            ghzc.GHZ_XYY(), ghzc.GHZ_XXX(), ghzc.GHZ_XYX()]


def csv_from_sorted(circuit_name, sorted_keys, sorted_counts):
    """Create csv string from sorted keys and counts"""
    print(circuit_name, end=';')
    for key in sorted_keys:
        print(key, end=';')
    print()
    print(datetime.datetime.now().isoformat(), end=';')
    for count in sorted_counts:
        print(count, end=';')
    print()


def csv(circuit_name, counts_exp):
    """Create csv string from circuit name and counts"""
    sorted_keys = sorted(counts_exp.keys())
    sorted_counts = []
    for key in sorted_keys:
        sorted_counts.append(counts_exp.get(key))
    csv_from_sorted(circuit_name, sorted_keys, sorted_counts)


# Iterate executing
for i in circuits:
    circuit_name = type(i).__name__
    if args.qasm:
        print(circuit_name)
        print(i.qasm())
    if args.draw:
        print(i.qc.draw())
    job_exp = execute(i.qc, backend=backend, shots=shots,
                      max_credits=max_credits)

    if args.quiet or args.to_err:
        job_monitor(job_exp, quiet=args.quiet,
                    output=sys.stderr if args.to_err else sys.stdout)
    else:
        job_monitor(job_exp)

    result_exp = job_exp.result()
    if args.aer:
        print(result_exp.get_statevector())
        # print(result_exp.get_unitary())
    counts_exp = result_exp.get_counts(i.qc)
    if args.results:
        print(counts_exp)
    csv(circuit_name, counts_exp)
    print()

verbosity('Done!', 1)

# End
