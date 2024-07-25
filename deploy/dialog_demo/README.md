## RUN 本地模型

```bash
MODELING_NAME=chatglm_v2 \
MODEL_NAME_OR_PATH=/media/zjin/Data/dataset/model/chatglm/6B_v2 \
streamlit run deploy/dialog_demo/chat.py
```

## RUN ChatGPT

```bash
MODELING_NAME=gpt-3.5-turbo \
streamlit run deploy/dialog_demo/chat.py
```

- MODELING_NAME: gpt-3.5-turbo 或 gpt-4