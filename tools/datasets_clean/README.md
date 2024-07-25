## 1.介绍

包含数据列表与数据处理脚本。
公司的数据集均放在公司NAS上， 关于公司的NAS参考[nas存储配置](https://alidocs.dingtalk.com/i/nodes/r1R7q3QmWe7Kv1ZPFmkn1ZQ4JxkXOEP2?doc_type=wiki_doc&dontjump=true&iframeQuery=utm_source%3Dportal%26utm_medium%3Dportal_recent#)。

## 2.数据列表

### 2.1 开源数据集

- [MNBVC超大规模中文语料集](docs/开源数据集/mnbvc.md)
- [THUCNews中文文本数据集](docs/开源数据集/thuc_news.md)
- [LCCC-中文对话语料集](docs/开源数据集/lccc.md)
- [CC100-chinese-中文语料集](docs/开源数据集/cc100_chinese.md)
- [小学题库25万条](docs/开源数据集/school_math_25w.md)
- [个性化角色对话40万条](docs/开源数据集/generated_chat_40w.md)
- [多轮对话80万条](docs/开源数据集/multi_chat_80w.md)
- [对话200万条](docs/开源数据集/chat_200w.md)


其他参考数据:
- [CLUECorpus2020](https://github.com/CLUEbenchmark/CLUECorpus2020)
- [悟道](https://data.baai.ac.cn/details/WuDaoCorporaText)
- [ChatGPT相关数据集](https://zhuanlan.zhihu.com/p/614269538)


### 3.数据处理脚本

- script/un_zip.py: 将加密的压缩包数据进行解压
