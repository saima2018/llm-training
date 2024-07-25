from .base_trainer import BaseTrainer


class AccelerateTrainer(BaseTrainer):
    def __init__(self, model, optimizer, criterion):
        super().__init__(model, optimizer, criterion)

    def train_one_batch(self, batch):
        # loss = self.criterion(model_output, targets)

        # loss.backward()
        # self.optimizer.step()
        pass
