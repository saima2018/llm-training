import argparse
import json

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--input_file",
    type=str,
    default="/media/zjin/Data/dataset/datas/baidu_baike/daxue.json",
    help="输入json路径",
)
parser.add_argument(
    "--output_file",
    type=str,
    default="/media/zjin/Data/dataset/datas/baidu_baike/daxue.txt",
    help="输出txt路径",
)
args = parser.parse_args()


def main():
    with open(args.input_file, "r") as fp:
        datas = json.load(fp)
    fp.close()

    txt_fp = open(args.output_file, "a", encoding="utf-8")
    for dat in datas[:100]:
        txt_str = dat["data"] + "\n\n"
        txt_fp.write(txt_str)
    txt_fp.close()
    return


if __name__ == "__main__":
    main()
