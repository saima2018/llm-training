FROM 192.168.1.248:10010/platform/train_env_cuda118_ubuntu22_py10:v0.1
# apt related libraries
ADD requirements.txt ./
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir
ADD . /workspace/

# rm extra
#RUN rm -rf /workspace/apis/***

WORKDIR /workspace

# docker starts running the default command
CMD ["/bin/bash"]