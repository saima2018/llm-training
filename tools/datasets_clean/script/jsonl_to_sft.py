import argparse
import json

import jsonlines

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--input_file",
    type=str,
    default="/media/zjin/Data/dataset/dataset/对话类/Generated_Chat_40W/data/generated_chat_0.4M__zh__396004.json",
    help="输入json路径",
)
parser.add_argument(
    "--output_file",
    type=str,
    default="/media/zjin/Data/dataset/datas/baidu_baike/generated_chat.jsonl",
    help="输出jsonl路径",
)
args = parser.parse_args()


def main():
    # 参考: https://jsonlines.readthedocs.io/en/latest/
    reader = jsonlines.open(args.input_file)
    writer = jsonlines.open(args.output_file, mode="w")

    for obj in reader:
        dat = json.loads(json.dumps(obj, ensure_ascii=False))
        input = dat["instruction"]
        if len(dat["input"]) > 0:
            input = f"{input}\n{dat['input']}"
        writer.write({"chat": [{"input": input, "output": dat["output"]}]})

    reader.close()
    writer.close()
    return


if __name__ == "__main__":
    main()
