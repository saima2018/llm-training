from .accelerate_trainer import AccelerateTrainer
from .deepspeed_trainer import DeepspeedTrainer
from .torchddp_trainer import TorchDistributeTrainer


class TrainerFactory:
    trainer_map = {
        "accelerate": AccelerateTrainer,
        "deepspeed": DeepspeedTrainer,
        "torch_dp": TorchDistributeTrainer,
    }

    @staticmethod
    def get_trainer(backends, **kwargs):
        TrainerClass = TrainerFactory.trainer_map[backends]
        return TrainerClass(**kwargs)
