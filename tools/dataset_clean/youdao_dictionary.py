from gevent import monkey

monkey.patch_all()
import datetime
import logging
import os

import gevent
import jsonlines
from gevent.pool import Pool
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options

# 创建一个logger对象
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 创建一个输出到控制台的handler，并设置级别和格式化器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

# 将handler添加到logger
logger.addHandler(console_handler)

# 记录当前时间
current_time = datetime.datetime.now()
logger.debug("Current time is %s", current_time)


# 判断字典中英文文本长度是否正常。不正常则舍弃这个对话。
def check_values(dictionary):
    for value in dictionary.values():
        # 舍弃的单词数为1的文本
        if len(value.split()) <= 1:
            return False
    return True


# 判断字典中是否存在包含指定子串的键
def check_key_contains(dictionary, substring):
    for key in dictionary.keys():
        if substring in key:
            return True
    return False


# 使用selenium对数据进行翻译，选择浏览器为Edge
def translate_text(text):
    try:
        options = Options()
        # 浏览器后台运行
        options.add_argument("--headless")
        # 可修改浏览器种类，安装相应的WebDriver即可
        driver = webdriver.Edge(options=options)
        driver.get("https://fanyi.youdao.com/index.html#/")
        # 最长等待时间
        driver.implicitly_wait(5)
        driver.find_element(By.ID, "js_fanyi_input").clear()
        driver.find_element(By.ID, "js_fanyi_input").send_keys(text)
        trans = driver.find_element(By.CSS_SELECTOR, "#js_fanyi_output *").text
        driver.find_element(By.ID, "js_fanyi_input").clear()
        logger.info("Translated text: %s", text)
        driver.quit()
        data = trans.strip()
        if data == "" or data == "翻译数据正在路上..." or data == "对不起，我正在学习该语种中":
            return ""
        else:
            return trans.strip()
    except Exception as e:
        logger.error("Failed to translate text: %s, error: %s", text, str(e))
        # driver.quit()
        return ""


# 对jsonl中的一个json进行处理
def read_translate_line(line):
    chats = []
    missing_chat = []
    for chat, value in line.items():
        for data in value:
            # 这个判断语句中check_values用于筛选值长度符合预期的字典，check_key_contains（）可将某些键的字典不进行筛选直接进入
            if (
                check_values(data)
                or check_key_contains(data, "dany")
                or check_key_contains(data, "霍金")
            ):
                new_data = {}
                for key, value in data.items():
                    out = translate_text(value)
                    # 文本不干净可能出现特殊情况，进行跳过
                    if len(out) > 0:
                        new_data[key] = out
                    else:
                        missing_chat.append(data)
                print(len(new_data.keys()) > 1)
                # 只有长度满足字典的数量才进行添加
                if len(new_data.keys()) > 1:
                    chats.append(new_data)
                else:
                    # 翻译失败数据，方便后续处理
                    missing_chat.append(new_data)
                    # missing_data.write(new_data)
    if len(chats) > 0:
        logger.info("Translated chat: %s", chats)
        trans_data.write({"chat": chats})
    if len(missing_chat) > 0:
        missing_data.write({"chat": missing_chat})


def process_dictory(num):
    tasks = []
    # 最大协程数
    pool = Pool(num)
    # print(input_file)
    with jsonlines.open(input_file, "r") as f:
        for line in f:
            task = pool.spawn(read_translate_line, line)
            tasks.append(task)
    gevent.joinall(tasks)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Translate JSONL files using YouDao Translate."
    )
    parser.add_argument(
        "input_dir", help="The directory containing the input JSONL files."
    )
    parser.add_argument(
        "output_dir", help="The directory to save the translated JSONL files."
    )
    parser.add_argument(
        "missing_dir", help="The directory to save the missing data JSONL files."
    )
    parser.add_argument(
        "--max-coroutines",
        type=int,
        default=16,
        help="The maximum number of greenlets to run simultaneously.",
    )
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    missing_dir = args.missing_dir
    max_coroutines = args.max_coroutines

    # 输入目录
    dictory = input_dir
    file_names = os.listdir(dictory)
    jsonl_files = [file for file in file_names if file.endswith("jsonl")]
    for file in jsonl_files:
        input_file = file
        # 输出目录
        out_path = os.path.join(output_dir, input_file)
        # 翻译失败目录
        missing_path = os.path.join(missing_dir, input_file)
        missing_data = jsonlines.open(missing_path, "w")
        trans_data = jsonlines.open(out_path, "w")
        input_file = os.path.join(dictory, input_file)
        process_dictory(max_coroutines)
