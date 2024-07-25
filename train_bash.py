from train.trainer import get_train_args, train_ppo, train_rm


def main():
    (
        model_args,
        data_args,
        training_args,
        finetuning_args,
        # general_args,
    ) = get_train_args()

    # if general_args.stage == "pt":
    #     run_pt(model_args, data_args, training_args, finetuning_args)
    # if finetuning_args.stage == "sft":
    #     run_sft(model_args, data_args, training_args, finetuning_args)
    if finetuning_args.stage == "rm":
        train_rm(model_args, data_args, training_args, finetuning_args)
    elif finetuning_args.stage == "ppo":
        train_ppo(model_args, data_args, training_args, finetuning_args)


if __name__ == "__main__":
    main()
