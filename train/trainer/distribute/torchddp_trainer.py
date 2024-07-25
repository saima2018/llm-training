from .base_trainer import BaseTrainer


class TorchDistributeTrainer(BaseTrainer):
    def __init__(self, model, optimizer, criterion):
        super().__init__(model, optimizer, criterion)
