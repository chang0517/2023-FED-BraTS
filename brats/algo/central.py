from base import BaseTrainer

class CentralTrainer(BaseTrainer):
    """
    Centralized training algorithm.
    """

    def __init__(self, config):
        super(CentralTrainer, self).__init__(config)

    def train(self):
        """
        Train the model.
        """
        for epoch in range(self.config['n_epochs']):
            self._train_epoch(epoch)

    def _train_epoch(self, epoch):
        """
        Train the model for one epoch.
        """
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.trainloader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.config['log_interval'] == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.trainloader.dataset),
                    100. * batch_idx / len(self.trainloader), loss.item()))
        self.lr_scheduler.step()

    def _build_model(self):
        """
        Build the model.
        """
        model = self.config.init_obj('arch', nn)
        return model

    def _build_dataloader(self):
        """
        Build the data loader.
        """
        train_loader = self.config.init_obj('data_loader', data)
        return train_loader

    def _build_optimizer(self):
        """
        Build the optimizer.
        """
        params = self.model.parameters()
        optimizer = self.config.init_obj('optimizer', torch.optim, params)
        return optimizer

    def _build_lr_scheduler(self):
        """
        Build the learning rate scheduler.
        """
        lr_scheduler = self.config.init_obj('lr_scheduler', torch.optim.lr_scheduler, self.optimizer)
        return lr_scheduler

    def _build_logger(self):
        """
        Build the logger.
        """
        logger = logging.getLogger(self.config['name'])
        logger.setLevel(self.config['log_level'])
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        if self.config['log_to_console']:
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        if self.config['log_to_file']:
            fh = logging.FileHandler(self.config['log_file'])
            fh.setFormatter(formatter)