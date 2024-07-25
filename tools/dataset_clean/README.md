
# 有道翻译脚本

## 运行条件
1.安装selenium以及相应浏览器的WebDriver

2.其余库pip安装即可

3.写数据延迟较久

## 使用方法
### 1.翻译目录中的所有jsonl文件
使用Youdao_dictionary.py

在命令行输入命令,包括输入文件路径，输出文件路径，翻译失败文件路径以及并发数量，默认为16。这三个需要存放于不同的文件夹中。
且输出文件名和翻译失败文件名一致。

`usage: Youdao_dictory.py [-h] [--max-coroutines MAX_COROUTINES]
                         input_dir output_dir missing_dir
Youdao_dictory.py: error: the following arguments are required: input_dir, output_dir, missing_dir`


### 2.翻译单个文本
`from Youdao_dictory import translate_text`

`translate_text(text)`,输入为要翻译的文本，返回值为翻译后的文本。如果返回为''则说明翻译失败，可能是网络的问题。
