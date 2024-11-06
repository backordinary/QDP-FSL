# https://github.com/gaqqie/gaqqie-door/blob/eae3c2a3ccf901f9853ea0ab2ddea9a1b851e52d/gaqqie_door/adaptor/qiskit/gaqqie_provider.py
from threading import Lock
from typing import List

from qiskit import Aer
from qiskit.providers import ProviderV1

from .gaqqie_backend import GaqqieBackend
from ...rest import Configuration, JobApi, DeviceApi, ProviderApi, Device
from ...rest.api_client2 import ApiClient2


class GaqqieProvider(ProviderV1):
    def __init__(
        self, name: str = "gaqqie", status: str = None, description: str = None
    ) -> None:
        """Initializes GaqqieProvider object.

        Parameters
        ----------
        name : str, optional
            provider name, by default "gaqqie".
        status : str, optional
            provider status, by default None.
        description : str, optional
            provider description, by default None.
        """
        super().__init__()
        self._name: str = name
        self._status: str = status
        self._description: str = description
        self._url: str = None
        self._api_client: ApiClient2 = None
        self._lock: Lock = Lock()

    def enable_account(self, url: str) -> None:
        """Enables account.

        Parameters
        ----------
        url : str
            the base URL of the gaqqie API for users.
        """
        # thread safe
        with self._lock:
            self._url = url
            self._init_api()

    def _init_api(self) -> None:
        rest_config = Configuration()
        rest_config.host = self._url

        self._api_client = ApiClient2(rest_config)
        self._job_api = JobApi(api_client=self._api_client)
        self._device_api = DeviceApi(api_client=self._api_client)
        self._provider_api = ProviderApi(api_client=self._api_client)

    @property
    def name(self) -> str:
        """Returns provider name.

        Returns
        -------
        str
            provider name.
        """
        return self._name

    @property
    def status(self) -> str:
        """Returns provider status.

        Returns
        -------
        str
            provider status.
        """
        return self._status

    @property
    def description(self) -> str:
        """Returns provider description.

        Returns
        -------
        str
            provider description.
        """
        return self._description

    @property
    def job_api(self) -> JobApi:
        """Returns "job interface" of the gaqqie API for users.

        Returns
        -------
        JobApi
            job interface.
        """
        return self._job_api

    @property
    def device_api(self) -> DeviceApi:
        """Returns "device interface" of the gaqqie API for users.

        Returns
        -------
        DeviceApi
            device interface.
        """
        return self._device_api

    @property
    def provider_api(self) -> ProviderApi:
        """Returns "provider interface" of the gaqqie API for users.

        Returns
        -------
        ProviderApi
            provider interface.
        """
        return self._provider_api

    def backends(self, name: str = None, **kwargs) -> List[GaqqieBackend]:
        """Returns backends.

        Parameters
        ----------
        name : str, optional
            backend name to get, by default None

        Returns
        -------
        List[GaqqieBackend]
            list of backends.
        """
        # get backend information from cloud
        backends = []
        if name is None:
            devices = self.device_api.get_devices()

            # create backend instances
            for device in devices:
                aer_backend = Aer.get_backend("qasm_simulator")
                config = aer_backend.configuration()
                config.backend_name = device.name
                config.n_qubits = device.num_qubits
                config.simulator = True
                config.max_shots = device.max_shots
                backend = GaqqieBackend(config, device, self)
                backends.append(backend)
        else:
            device = self.device_api.get_device_by_name(name)

            # create backend instance
            aer_backend = Aer.get_backend("qasm_simulator")
            config = aer_backend.configuration()
            config.backend_name = device.name
            config.n_qubits = device.num_qubits
            config.simulator = True
            config.max_shots = device.max_shots
            backend = GaqqieBackend(config, device, self)
            backends.append(backend)

        return backends

    def providers(self, name: str = None, **kwargs) -> List["GaqqieProvider"]:
        """Returns providers.

        Parameters
        ----------
        name : str, optional
            provider name to get, by default None

        Returns
        -------
        List[GaqqieProvider]
            list of providers.
        """
        # get backend information from cloud
        gaqqie_providers = []
        if name is None:
            providers = self.provider_api.get_providers()

            # create provider instances
            for provider in providers:
                gaqqie_provider = GaqqieProvider(
                    name=provider.name,
                    status=provider.status,
                    description=provider.description,
                )
                gaqqie_providers.append(gaqqie_provider)
        else:
            provider = self.provider_api.get_provider_by_name(name)

            # create backend instance
            gaqqie_provider = GaqqieProvider(
                name=provider.name,
                status=provider.status,
                description=provider.description,
            )
            gaqqie_providers.append(gaqqie_provider)

        return gaqqie_providers
