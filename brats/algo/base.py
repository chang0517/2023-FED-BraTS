import random
from abc import ABCMeta, abstractmethod

import numpy as np
import torch


class BaseTrainer(metaclass=ABCMeta):
    """
    Base class for supervised training
    The following methods are abstract methods, which must be implemented in the subclass.
    """

    def __init__(self, config):
        self.config = config

        self._set_random_seed()
        self._build_model()
        self._build_dataloader()
        self._build_optimizer()
        self._build_lr_scheduler()
        self._build_logger()

    @abstractmethod
    def train(self):
        ...

    def _set_random_seed(self):
        seed = self.config.get('seed', 0)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    @abstractmethod
    def _build_model(self):
        ...

    @abstractmethod
    def _build_dataloader(self):
        ...

    @abstractmethod
    def _build_optimizer(self):
        ...

    @abstractmethod
    def _build_lr_scheduler(self):
        ...

    @abstractmethod
    def _build_logger(self):
        ...


class BaseFederatedTrainer(BaseTrainer):
    """
    Base class for federated learning algorithms.
    Any class inheriting from this class must implement client selection, local training, and aggregation.
    """

    def __init__(self, config):
        super(BaseFederatedTrainer, self).__init__(config)
        self.initialized = False

    def train(self):
        assert self.initialized, 'The trainer is not initialized.'
        for client_id in self._client_selection():
            self._local_train(
                client_id, self.config.local_epochs,
                self.train_loaders[client_id],
                self.models[client_id],
                self.criteria[client_id],
                self.optimizers[client_id]
            )
        self._aggregate()

    @abstractmethod
    def _local_train(self, client_id, n_epochs, trainloader, model, criterion, optimizer):
        ...

    @abstractmethod
    def _client_selection(self):
        ...

    @abstractmethod
    def _local_training(self):
        ...

    @abstractmethod
    def _aggregate(self):
        ...

    def _build_model(self):
        ...

    def _build_dataloader(self):
        ...

    def _build_optimizer(self):
        ...

    def _build_lr_scheduler(self):
        ...

    def _build_logger(self):
        ...


class BaseIncrementalTrainer(BaseTrainer):
    """
    Base class for incremental learning algorithms.
    IIL(Institutional Incremental Learning) and CIIL(Cyclic Institutional Incremental Learning) are implemented.
    """

    def __init__(self, config):
        super(BaseIncrementalTrainer, self).__init__(config)

    def train(self):
        ...

    @abstractmethod
    def _populate(self):
        ...

    @abstractmethod
    def _aggregate(self):
        ...
