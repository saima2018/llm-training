# ===================================
# logger
# 使用方法: add_file_handler_to_logger后使用logger.info、logger.warning、logger.error即可在输出log到屏幕的同时输出到log文件
# ===================================
import datetime
import os
import sys

from loguru import logger as loguru_logger

loguru_logger.configure(
    handlers=[
        dict(
            sink=sys.stderr,
            filter=lambda record: record["extra"]["console"],
            level="INFO",
        )
    ]
)
logger = loguru_logger.bind(console=True)
prefix = datetime.datetime.now().strftime("%Y%m%d%H%M%S")


def log_rank0(msg, rank):
    if rank <= 0:
        logger.info(msg)


def add_file_handler_to_logger(name: str, dir_path: str = "", level="DEBUG"):
    """增加logger写入
    :param name: log文件名
    :param dir_path: log文件地址
    :param level: 写入log等级
    :return:
    """

    loguru_logger.add(
        sink=os.path.join(dir_path, f"{name}-{prefix}-{level}.log"),
        level=level,
        rotation="1 day",
        retention="7 days",
        filter=lambda record: "tracker" not in record["extra"]
        and "console" in record["extra"]
        and record["level"] != "ERROR"
        if level != "ERROR"
        else True,
        enqueue=True,
    )
