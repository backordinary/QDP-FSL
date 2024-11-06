# https://github.com/jxerome/quantum-book/blob/75d13679d7cddcae2e2b9cf1a53e570fbaac7358/set-ibm-account.py
#!/usr/bin/env python

import qiskit as q
from getpass import getpass

jeton = getpass("Indiquez votre jeton d'API IBM: ")
q.IBMQ.save_account(jeton)
