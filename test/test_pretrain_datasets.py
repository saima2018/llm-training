from data import PretrainDataset
from modeling import ModelingFactory

data_dir = "../data_samples/pretrain"
model_path = "/media/nas/data/model/base_model/chatglm/6B_v2"
modeling = ModelingFactory.get(
    modeling_name="chatglm_v2", model_name_or_path=model_path
)
train_dataset = PretrainDataset(
    train_folder=data_dir, tokenizer=modeling.tokenizer, max_length=1024
)
