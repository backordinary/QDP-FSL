# https://github.com/silkycove/qfluxcap/blob/89a6bf44bc688075f9c6ed705927409482398186/settoken.py
# settoken.py Copyright (c) 2021 Masami Yamakawa
# SPDX-License-Identifier: MIT
#
from qiskit import IBMQ
import getpass

token = getpass.getpass('IBM Q Experience Token:')
IBMQ.save_account(token, overwrite=True)
print('The credential has been saved:', token[:4],'***',token[-4:])
