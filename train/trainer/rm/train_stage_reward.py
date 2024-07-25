from transformers import Seq2SeqTrainingArguments

from data import get_dataset, preprocess_dataset, split_dataset
from train.hyperparams import DataArguments, FinetuningArguments, ModelArguments
from train.trainer.common import load_model_and_tokenizer
from train.trainer.rm.collator import PairwiseDataCollatorWithPadding
from train.trainer.rm.rm_peft_trainer import PairwisePeftTrainer
from train.trainer.utils.ploting import plot_loss


def train_rm(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: Seq2SeqTrainingArguments,
    finetuning_args: FinetuningArguments,
):
    dataset = get_dataset(model_args, data_args)
    model, tokenizer = load_model_and_tokenizer(
        model_args, finetuning_args, training_args.do_train, stage="rm"
    )
    dataset = preprocess_dataset(
        dataset, tokenizer, data_args, training_args, stage="rm"
    )
    data_collator = PairwiseDataCollatorWithPadding(tokenizer)

    training_args.remove_unused_columns = False  # 适用于pairwise成对数据集的设置

    # 初始化训练器
    trainer = PairwisePeftTrainer(
        finetuning_args=finetuning_args,
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        **split_dataset(dataset, data_args.dev_ratio, training_args.do_train)
    )

    # 训练
    if training_args.do_train:
        train_result = trainer.train()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        trainer.save_model()
        if trainer.is_world_process_zero() and model_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # 验证
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 预测
    if training_args.do_predict:
        predict_results = trainer.predict(dataset, metric_key_prefix="predict")
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)
