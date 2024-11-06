# https://github.com/avkhadiev/interpolators/blob/3911f7f5f03e9a5e2c5bd05d278a73fcb2cbe59a/optimizer.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Defines SPSA optimizer

from qiskit import *
from qiskit.aqua.components.optimizers import SPSA

max_trials = 500
optimizer = qiskit.aqua.components.optimizers.SPSA(max_trials)

if __name__ == '__main__':
    print(optimizer.setting)

