# MNBVC超大规模中文语料集

## 1.存储

- 数据来源: https://github.com/esbatmop/MNBVC
- 公司NAS目录: dataset/MNBVC

## 2.说明

MNBVC(Massive Never-ending BT Vast Chinese corpus)是一个超大规模中文语料集，
截止到2023.04.19已完成2.7T数据的更新，仍然在持续收集中...

存储目录如下:

```
MNBVC
├── data
│   ├── 20221224.zip
│   ├── 20221225.zip
│   ├── 20230101.zip
...
│   ├── 20230304.zip
│   └── 20230305.zip
└── README.md
```

- README.md: 对数据来源和压缩包密码进行了说明
- data: data目录为存储的数据，需要自行解压使用，密码见README.md

备注: 当前数据集还没有进行清洗，不一定适合模型训练使用

