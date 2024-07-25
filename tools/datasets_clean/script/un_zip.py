# =================================================================
# 本脚本采用7z命令对加密的zip包进行自动解压，7z解压速度较其他方案更快
# =================================================================

import argparse
import os
import subprocess

parser = argparse.ArgumentParser(description="数据解压代码")
parser.add_argument("--data_name", type=str, default="MNBVC", help="输入数据")
parser.add_argument(
    "--input_dir",
    type=str,
    default="/media/zjin/Data/dataset/datas/MNBVC/data",
    help="输入目录",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="/media/zjin/Data/dataset/datas/MNBVC/un_zip",
    help="输出目录",
)
parser.add_argument("--password", type=str, default="253874", help="password")
args = parser.parse_args()


def main():
    input_dir, output_dir, password = args.input_dir, args.output_dir, args.password
    os.makedirs(output_dir, exist_ok=True)

    print("developing...")
    files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)]
    length = len(files)
    for i, zip_file in enumerate(files):
        try:
            print(f"unzip {i}/{length}")
            name = zip_file.replace(".zip", "").split("/")[-1]
            if name in os.listdir(output_dir):
                print(f"{name} is in {output_dir}...")
                continue
            # sudo apt install p7zip-full p7zip-rar
            subprocess.call(
                [
                    "7z",
                    "x",
                    "-p{}".format(password),
                    zip_file,
                    "-o{}".format(output_dir),
                ]
            )

            print(f"finished unzip {zip_file}...")
        except:
            print(f"{zip_file} failed")
    return


if __name__ == "__main__":
    main()
