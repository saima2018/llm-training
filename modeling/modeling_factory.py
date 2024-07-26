from .baichuan.baichuan_modeling import ModelBaichuan
from .baichuan_chat.baichuan_chat_modeling import ModelBaichuanChat
from .chatglm_v2.chatglm_v2_modeling import ModelChatGLMV2
from .chatgpt.chatgpt_modeling import ModelChatGPT
from .gpt2.gpt2_modeling import ModelGPT2
from .llama.llama_modeling import ModelLLaMa
from .llama3_classification.llama3_classification_modeling import ModelLLaMa3Classification
from .moss.moss_modeling import ModelMoss


# Factory class to provide access of all models
class ModelingFactory:
    factory_map = {
        "chatgpt": ModelChatGPT,
        "gpt2": ModelGPT2,
        "llama": ModelLLaMa,
        "llama3_classification": ModelLLaMa3Classification,
        "moss": ModelMoss,
        "baichuan": ModelBaichuan,
        "baichuan_chat": ModelBaichuanChat,
        "chatglm_v2": ModelChatGLMV2,
    }

    @staticmethod
    def get(modeling_name: str, model_name_or_path: str):  # initialise model class with model name
        """为了统一输入，当前采用model_path而非**kwargs"""
        Modeling = ModelingFactory.factory_map[modeling_name]
        return Modeling(model_name_or_path=model_name_or_path)

    @staticmethod
    def list():  # 输出支持的模型列表
        return list(ModelingFactory.factory_map.keys())
