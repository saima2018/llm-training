#!/usr/bin/env bash

build_tag=llm_model


# 停止运行的docker 服务
docker rm -f $build_tag
echo "stop container $build_tag"

# 生成容器
docker build -t $build_tag ./ --network host -f ./docker/cu118-ubuntu22.04-base/Dockerfile

echo "Built image $build_tag"

