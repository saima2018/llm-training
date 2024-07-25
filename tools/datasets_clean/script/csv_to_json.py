# =================================================================
# 本脚本将csv文件转为json进行存储，存储为list格式，list中包含所有样本
# 每个样本为dict，表头为key，对应位置的值为value
# =================================================================

import argparse
import csv
import json

parser = argparse.ArgumentParser(description="csv数据转json")
parser.add_argument(
    "--input_csv_file",
    type=str,
    default="/media/zjin/Data/dataset/dataset/未处理/法律/lawzhidao_filter.csv",
    help="输入csv文件",
)
parser.add_argument(
    "--output_json_file",
    type=str,
    default="/media/zjin/Data/dataset/dataset/专业数据分类/法律/lawzhidao_filter.json",
    help="输出json文件",
)
args = parser.parse_args()


def read_csv(csv_file):
    with open(csv_file) as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        data = [headers]
        for row in f_csv:
            data.append(row)
    return data


def main():
    print(f"conv {args.input_csv_file} to {args.output_json_file}")
    # 读取表格
    data = read_csv(args.input_csv_file)
    # 表格转为json列表
    out_data = []
    heads = data[0]
    for dat_list in data[1:]:
        dat = dict()
        for i, head in enumerate(heads):
            dat[head] = dat_list[i]
        out_data.append(dat)

    # 保存json文件
    with open(args.output_json_file, "w") as fp:
        json.dump(out_data, fp, ensure_ascii=False)
    print(f"finish {args.input_csv_file} to {args.output_json_file}")
    return


if __name__ == "__main__":
    main()
