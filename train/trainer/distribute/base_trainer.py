class BaseTrainer:
    def __init__(self, model, tokenizer, optimizer, lr_scheduler, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def save_checkpoint(self, model_path, tag):
        raise NotImplementedError("Subclasses must implement save_checkpoint()")

    def load_checkpoint(self, model_path):
        raise NotImplementedError("Subclasses must implement load_checkpoint()")
