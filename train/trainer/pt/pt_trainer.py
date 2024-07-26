import torch

from train.trainer.distribute.base_trainer import BaseTrainer
from utils import logger


class PretrainSFTTrainer:
    def __init__(self, ddp_trainer: BaseTrainer):
        self.model = ddp_trainer.model_engine
        self.tokenizer = ddp_trainer.tokenizer
        self.global_rank = torch.distributed.get_rank()
        self.ddp_trainer = ddp_trainer
        self.args = ddp_trainer.args

    def train(self, dataloader, epochs):
        steps_per_epoch = len(dataloader)
        total_steps = epochs * steps_per_epoch
        trained_steps = 0

        for epoch in range(epochs):
            self.model.train()

            epoch_loss = 0
            for step, batch in enumerate(dataloader):
                ids = batch[0].to(self.model.device)
                labels = batch[1].to(self.model.device)
                # batch = batch.to(self.model.device)
                loss = self.model(input_ids=ids, labels=labels)[0]
                self.model.backward(loss)
                self.model.step()
                trained_steps += 1

                trained_loss = loss.item()
                epoch_loss += trained_loss
                mean_loss = epoch_loss / (step + 1)

                self.print_rank_0(
                    f"Epoch:{epoch}  step:{step}/{steps_per_epoch} Trained_Step:{trained_steps}/{total_steps} "
                    f"mean_loss:{mean_loss} loss:{trained_loss}",
                    self.global_rank,
                )

                if self.args.save_step_interval:
                    if trained_steps % self.args.save_step_interval == 0:
                        self.print_rank_0(
                            f"saving checkpoints trained_steps: {trained_steps}",
                            self.global_rank,
                        )
                        self.save_checkpoint(
                            model_path=self.args.checkpoint_dir, tag=f"{trained_steps}"
                        )
                        self.print_rank_0(
                            f"saving dataloader trained_steps: {trained_steps}",
                            self.global_rank,
                        )
                        torch.save(
                            dataloader,
                            f"{self.args.checkpoint_dir}/dataloader_state_{trained_steps}.pt",
                        )

            # save epoch
            if not self.args.save_step_interval:
                self.print_rank_0(
                    f"saving checkpoints epoch: {epoch}", self.global_rank
                )
                self.save_checkpoint(model_path=self.args.checkpoint_dir, tag=epoch)
                self.print_rank_0(f"saving dataloader epoch: {epoch}", self.global_rank)
                torch.save(
                    dataloader,
                    f"{self.args.checkpoint_dir}/dataloader_state_{epoch}.pt",
                )

    def print_rank_0(self, msg, rank=0):
        if rank <= 0:
            logger.info(msg)

    def to_device(batch, device):
        output = {}
        for k, v in batch.items():
            try:
                output[k] = v.to(device)
            except:
                output[k] = v
        return output

    # 其他函数
    def evaluate(self):
        ...

    def save_checkpoint(self, **kwargs):
        self.ddp_trainer.save_checkpoint(**kwargs)
