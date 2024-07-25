import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from data import SFTDataset
from modeling import ModelingFactory

data_dir = "../data_samples/sft"
model_path = "/media/zjin/Data/dataset/model/chatglm/6B_v2"
modeling = ModelingFactory.get(
    modeling_name="chatglm_v2", model_name_or_path=model_path
)
train_dataset = SFTDataset(
    data_dir=data_dir,
    tokenizer=modeling.tokenizer,
    model_conf=modeling.config,
    token_encode_function=modeling.token_encode_sft,
    max_data_input_length=1000,
)
