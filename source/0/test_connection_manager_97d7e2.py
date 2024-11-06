# https://github.com/QuentinPrigent/quantum-computing-project/blob/499f611cdd9024c8ae424cd60c4e67a7db4a7529/test/services/test_connection_manager.py
import unittest
from unittest.mock import MagicMock
from qiskit import IBMQ

from src.services.connection_manager import account_initialization_manager


class TestAccountInitializationManager(unittest.TestCase):
    def test_account_initialization_manager_with_no_key(self):
        IBMQ.save_account = MagicMock()
        IBMQ.load_account = MagicMock()
        IBMQ.stored_account = MagicMock(return_value={})
        account_initialization_manager()
        self.assertTrue(IBMQ.save_account.called)
        self.assertTrue(IBMQ.load_account.called)

    def test_account_initialization_manager_with_a_key(self):
        IBMQ.save_account = MagicMock()
        IBMQ.load_account = MagicMock()
        IBMQ.stored_account = MagicMock(return_value={"token": "api-token"})
        account_initialization_manager()
        self.assertFalse(IBMQ.save_account.called)
        self.assertTrue(IBMQ.load_account.called)


if __name__ == '__main__':
    unittest.main()
