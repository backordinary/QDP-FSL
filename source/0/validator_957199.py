# https://github.com/albaaparicio12/TFG-App/blob/38a589b4d9a96d0d71fe61cdd89093d2ce827021/src/business/base/Validator.py
from http.client import HTTPException

from qiskit import IBMQ
from qiskit import Aer
from qiskit.providers.ibmq import IBMQAccountCredentialsInvalidUrl, IBMQProviderError, IBMQAccountError


class Validator:

    @staticmethod
    def check_local_device(device: str):
        """
        Comprueba que el simulador introducido por parámetro existe y es válido.
        :param device: nombre del simulador seleccionado.
        :return: InvalidValueException en el caso de que no exista o sea inválido.
        """
        if device not in [backend._options.get('method') for backend in Aer.backends()
                          if backend._options.get('method') != None] or device == 'stabilizer' \
                or device == 'extended_stabilizer':
            raise InvalidValueException("The specified local device does not exist.", 2000)

    @staticmethod
    def check_token(token: str):
        """
        Comprueba que el token introducido por parámetro pertenece a una cuenta válida.
        :param token: token del usuario a comprobar.
        :return: True si es válido o ya existe en el sistema, False en caso contrario.
        """
        try:
            IBMQ.enable_account(token)
            return True
        except IBMQAccountCredentialsInvalidUrl or IBMQProviderError:
            return False
        except IBMQAccountError:
            return True

    @staticmethod
    def check_n_executions(n_executions: int):
        """
        Comprueba que el número de ejecuciones es un número entre 0 y 20000.
        :param n_executions: número de ejecuciones introducidas por el usuario.
        :return: InvalidValueException si el valor introducido se encuentra fuera del rango establecido.
        """
        if n_executions <= 0 or n_executions >= 20000:
            raise InvalidValueException("The specified number of executions is invalid.", 2000)


class InvalidTokenException(HTTPException):
    def __init__(self, m, code):
        self.args = m
        self.message = m
        self.errors = code


class InvalidValueException(HTTPException):
    def __init__(self, m, code):
        self.args = m
        self.message = m
        self.errors = code
