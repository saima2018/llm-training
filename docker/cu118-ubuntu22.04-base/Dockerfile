FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# apt related libraries
RUN apt update && apt install -y python3 python3-pip wget vim git curl unzip build-essential libssl-dev libffi-dev

WORKDIR /workspace
# Install python libraries
RUN pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple
WORKDIR /workspace
RUN pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
ADD requirements.txt ./
RUN pip install -r requirements.txt -i https://pypi.douban.com/simple --no-cache-dir
ADD . /workspace/
#RUN pip install poetry && poetry config virtualenvs.create false
#RUN poetry install --no-interaction --no-ansi

# Timezone
ENV TZ=Asia/Shanghai
COPY ./zoneinfo /usr/share/zoneinfo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
# rm extra
#RUN rm -rf /workspace/apis/***

WORKDIR /workspace

# Docker starts and runs the default command
CMD ["/bin/bash"]