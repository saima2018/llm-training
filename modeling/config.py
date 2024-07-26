class ModelConfig:
    """
    Provides model configs
    """

    def __init__(
        self,
        max_model_input_length: int,
        pad_token_id: int,
        user_token: str,
        assistant_token: str,
        eos_token: str,
    ):
        self.max_model_input_length = max_model_input_length  # 模型最大输入token长度
        self.pad_token_id = pad_token_id  #  用于dataset中的batch数据构建-collate_fn
        self.user_token = (
            user_token  # 用户判别字符, _token为encode前的字符串， _token_id为encode后的token id
        )
        self.assistant_token = assistant_token  # AI判别字符
        self.eos_token = eos_token  # 结束字符

    # TODO: 加载模型时logger输出config
