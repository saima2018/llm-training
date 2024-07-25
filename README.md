# 1.介绍

## 1.1 文档

- [LLM训练框架-训练代码构建与设计](https://eq4whigzcp7.feishu.cn/docx/AP1MdbAxuoxKBmx0994c7dYAnLc)

# 1.2 代码架构说明

算法开发代码，具体部分文件夹说明:

- datasets: 数据处理部分，包含pretrain，sft，rlhf数据
- modeling: 模型部分，包含大模型架构和提供的方法
- training_scripts: 训练脚本，用于模型训练调用
- common: logger等公共实现
- config: 每个模型对应的配置选项,yaml和json
- deploy: 模型封装api对外发布
- tools: 包含评测代码和提供的chat前端
- docs: 文档


# 2.docker-compose训练

## 2.1 代码clone

```bash
git clone git@codeup.aliyun.com:62da4e731dbb334b98ba96bf/algo/model/llm_model.git
```

## 2.2 docker-compose

根据wiki文档配置好docker-compose中的environment。

```bash
docker-compose up -d    # 后台运行，想在前台运行可以不用加 -d
docker-compose logs -f  # 查看运行log
docker-compose down     # 停止运行
```

## 2.3 训练参数合并

进入模型保存目录，执行如下命令就和合并参数为pytorch_model.bin

```bash
./zero_to_fp32.py . pytorch_model.bin
```

将训练出来的参数替换掉原模型文件夹中的参数即可进行测试，注意测试不要覆盖原基础model，可以copy到自己目录

# 3.模型推理Demo测试

配置好docker-compose-demo.yml中参数:

- LOG_DIR: log日志地址
- MODELING_NAME: modeing名，现已支持chatglm_v2(需要考虑历史对话太长处理)，baichuan(需要考虑多轮对话)
- MODEL_NAME_OR_PATH: 模型地址

之后如下命令运行:

```bash
docker-compose -f docker-compose-demo.yml up -d    # 后台运行，想在前台运行可以不用加 -d
docker-compose -f docker-compose-demo.yml logs -f  # 查看运行log
docker-compose -f docker-compose-demo.yml down     # 停止运行
```


# 4.Develop

## 4.1 开发环境

```bash
virtualenv dev_env  # 没有virtualenv时需要安装 pip install virtualenv
source dev_env/bin/activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 4.2 模型训练

配置好train.sh中的具体变量，具体参数见: [LLM训练框架-训练代码构建与设计](https://eq4whigzcp7.feishu.cn/docx/AP1MdbAxuoxKBmx0994c7dYAnLc)。
运行如下命令开始训练:

```bash
bash run_train.sh
```

当前可训练模型:

- chatglm_v2
- baichuan_chat(13B)
- baichuan(7B)

支持的训练方式:
- Pretrain (full, lora)
- SFT (full, lora)
- Reward Model (lora)
- RLHF (lora)

## 4.3 demo部署

```bash
bash run_demo.sh
```

- MODELING_NAME: 模型名，当前支持gpt-3.5-turbo/gpt-4/chatglm_v2/baichuan_chat/baichuan
- MODEL_NAME_OR_PATH: 模型参数地址
- LOG_DIR: log输出地址

## 4.4 客观评测

配置好config中的evaluation.yaml的参数，之后运行如下命令开始评测:

```bash
bash run_evaluation.sh
```

评测后将在输出文件夹中输出对应评测指标文件。

# 5.规范
```bash
#防止不同formatter的格式差异，设置了precomit
#需在本地先初始化pre-commit环境，再进行commit
pip install pre-commit
pre-commit install
```
