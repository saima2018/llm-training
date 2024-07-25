import os
import sys
import streamlit as st

# 将目录设置到项目目录层再导入项目中的库
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from deploy.dialog_demo.streamlit_web import Sidebar, ChatWeb
from utils.logger import add_file_handler_to_logger, logger
from deploy.dialog_demo.llm_infer import LLMInfer

@st.cache_resource
def init_log_load_model():
    log_dir = os.getenv("LOG_DIR", default="./log")
    llm_infer_url = os.getenv("LLM_INFER_URL", default="http://127.0.0.1")
    llm_infer_port = os.getenv("LLM_INFER_PORT", default=9009)
    add_file_handler_to_logger(dir_path=log_dir, name="chat")  # 增加训练log输出
    text_llm_infer = LLMInfer(url=llm_infer_url, port=llm_infer_port)
    return text_llm_infer


def main():
    # 1. log设置 模型加载
    logger.info("start init_log_load_model")
    text_llm_infer = init_log_load_model()

    # 2.加载前端页面
    logger.info("start load streamlit")
    # 可扩充角色数量
    sidebar = Sidebar(role_list=(["AI"]))
    prompts = {
        "AI": f"",
    }
    init_messages = {
        "AI": f"您好，我是星凡AI机器人，你可以问我任何问题",
    }

    ChatWeb(
        text_llm_infer=text_llm_infer,
        sidebar=sidebar,
        prompt=prompts[sidebar.role],
        init_message=init_messages[sidebar.role],
    )


# LLM_INFER_URL=http://36.111.142.249 LLM_INFER_PORT=9008 streamlit run deploy/dialog_demo/chat.py --server.port 8051
if __name__ == "__main__":
    main()
