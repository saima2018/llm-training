# LCCC-中文对话语料集

## 1.存储

- 数据来源: https://huggingface.co/datasets/silver/lccc/tree/main
- 公司NAS目录: dataset/LCCC

## 2.说明

lccc是一套来自于中文社交媒体的对话数据。

数据对话样例如下:
```
[
    [
    "我 饿 了 。",
    "去 相 机 家 里 吃 … …",
    "相 机 今 年 木 有 回 去 T . T"
    ],
    [
    "网 络 大 实 话 里 说 的 是 也 许 你 能 在 网 络 里 找 到 你 想 要 的 友 情 但 永 远 不 会 找 到 你 想 要 的 爱 情",
    "你 过 来 我 们 什 么 关 系"
    ],
...
]
```

数据目录如下:

```
LCCC/
├── data
│   ├── lccc_base_test.jsonl.gz
│   ├── lccc_base_train.jsonl.gz
│   ├── lccc_base_valid.jsonl.gz
│   └── lccc_large.jsonl.gz
└── README.md
```

- README.md: 备注了数据来源
- data: 存储的数据，979M，需要自行解压使用