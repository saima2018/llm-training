# =================================================================
# 统计文件夹或文件中样本的个数
# 统计的是所有json文件中样本的数量，每个json文件存储的为列表格式
# =================================================================

import argparse
import json
import os

parser = argparse.ArgumentParser(description="统计文件夹或文件中样本的个数")
parser.add_argument(
    "--input_path",
    type=str,
    default="/media/zjin/Data/dataset/dataset/专业数据分类/法律/lawzhidao_filter.json",
    help="json文件或包含json文件的文件夹地址",
)
args = parser.parse_args()


def sum_json_samples(json_file):
    if not json_file.endswith(".json"):
        print(f"{json_file} 不是json文件，不进行统计")
        return 0
    with open(json_file, "r") as fp:
        data = json.load(fp)
    return len(data)


def main():
    print(f"统计{args.input_path}中json文件(list存储)包含的样本数量")
    num = 0
    if os.path.isfile(args.input_path):
        print("输入地址是文件")
        num = sum_json_samples(args.input_path)
    elif os.path.isdir(args.input_path):
        print("输入地址是文件夹")
        for root, _, files in os.walk(args.input_path):
            for f in files:
                num += sum_json_samples(os.path.join(root, f))
    else:
        print(f"输入必须是文件夹或文件，请检查你的输入地址:{args.input_path}")
    print(f"统计总样本数量为:{num}")
    return


if __name__ == "__main__":
    main()
