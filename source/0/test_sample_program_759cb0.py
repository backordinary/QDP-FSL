# https://github.com/iuliazidaru/qiskit-runtime-demo/blob/9f3b70b5b24a96d4912e00e922f147abbca27788/test/program/test_sample_program.py
# This code is part of qiskit-runtime.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the sample_program."""

import json
from fake_user_messenger import FakeUserMessenger
from unittest import TestCase
from qiskit.providers.aer import AerSimulator
from qiskit.providers.ibmq.runtime.utils import RuntimeEncoder, RuntimeDecoder
import qiskit_runtime_demo.program.sample_program as sample_program


class TestSampleProgram(TestCase):
    """Test sample_program."""

    def setUp(self) -> None:
        """Test case setup."""
        self.backend = AerSimulator()
        user_messenger = FakeUserMessenger()
        self.user_messenger = user_messenger

    def test_sample_program(self):
        """Test sample program."""
        inputs = {"iterations": 4}
        serialized_inputs = json.dumps(inputs, cls=RuntimeEncoder)
        unserialized_inputs = json.loads(serialized_inputs, cls=RuntimeDecoder)
        sample_program.main(self.backend, self.user_messenger, **unserialized_inputs)
        self.assertEqual(self.user_messenger.call_count, inputs["iterations"])
