# THUCNews中文文本数据集

## 1.存储

- 数据来源： http://thuctc.thunlp.org/
- 公司NAS目录: dataset/THUCNews


## 2.说明

THUCNews是由清华大学自然语言处理实验室推出的中文文本语料数据，包含了多个领域的数据。

数据存储目录如下:

```
THUCNews
├── data
│ └── THUCNews.zip
└── README.md
```

- README.md: 备注了数据来源
- data/THUCNews.zip: 存储的数据，压缩包1.6GB，需要自行解压使用

其数据THUCNews.zip解压后目录结构如下:

```
THUCNews
├── 体育
├── 娱乐
├── 家居
├── 彩票
├── 房产
├── 教育
├── 时尚
├── 时政
├── 星座
├── 游戏
├── 社会
├── 科技
├── 股票
└── 财经
```

备注: 当前数据集还没有进行清洗，不一定适合模型训练使用

