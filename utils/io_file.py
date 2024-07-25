import os.path
from typing import List

from utils.logger import logger


def find_files_from_extension(data_dir: str, extensions: List[str]) -> List[str]:
    """在输入目录下找出所有满足后缀名的文件，构成list返回
    :param data_dir: 输入目录
    :param extensions: 后缀名列表
    :return:
    """
    find_files: List[str] = list()
    if not os.path.isdir(data_dir):
        logger.error(f"{data_dir} is not existed")
        return find_files

    for root, dir, files in os.walk(data_dir):  # 遍历所有文件
        for name in files:
            extension = os.path.splitext(name)[-1]  # 获取文件后缀名
            if extension in extensions:  # 判断后缀名是否符合条件
                find_files.append(os.path.join(root, name))
    return find_files
