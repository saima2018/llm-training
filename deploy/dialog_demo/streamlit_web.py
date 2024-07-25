import os
import sys
import streamlit as st #
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from modeling.interface import ModelPredictRequest
from deploy.dialog_demo.llm_infer import LLMInfer
from utils import logger

# @torch.inference_mode()
def generate_interactive(text_llm_infer: LLMInfer, request: ModelPredictRequest):
    while True:
        logger.info(f"modeling input: "
                    f"temperature: {request.temperature} "
                    f"top_p: {request.top_p} "
                    f"top_k: {request.top_k} "
                    f"repetition_penalty: {request.repetition_penalty} "
                    f"max_new_tokens: {request.max_new_tokens} "
                    f"prompt: {request.prompt}\n"
                    f"history_messages: {request.history_messages}\n"
                    f"input_message: {request.input_message}")

        response, model_history = text_llm_infer.chat(request)
        logger.info(f"modeling output: {response}")
        yield response, model_history
        if len(response) >= 0:
            break

class Sidebar:
    def __init__(self, role_list):
        with st.sidebar:
            st.title("AI chatbot")
            self.role = st.radio("**choose role:**", role_list)

            if st.button("clear chat"):
                st.session_state.messages[self.role] = []

            add_vertical_space(3)

            st.markdown("### params:")
            self.model_request = ModelPredictRequest()
            param_container = st.container()
            with param_container:
                self.model_request.temperature = st.slider("Temperature", min_value=0.01, max_value=1.0, value=0.35)
                self.model_request.top_p = st.slider("Top P", min_value=0.01, max_value=1.0, value=0.9)
                self.model_request.top_k = st.slider("Top K", min_value=1, max_value=100, value=8)
                self.model_request.repetition_penalty = st.slider("Repetition penalty", min_value=1.0, max_value=100.0, value=1.0)
                self.model_request.max_new_tokens = st.slider("Max new tokens", min_value=1, max_value=20000, value=2048)
            add_vertical_space(3)
            st.write("Made with ❤️ by AI Team")
        return


def combine_model_predict_request(session_messages: list, input_message: str,  prompt: str, model_request: ModelPredictRequest) -> ModelPredictRequest:
    req: ModelPredictRequest = model_request.copy()
    req.prompt = prompt
    req.input_message = input_message
    req.history_messages = session_messages
    return req


class ChatWeb:
    def __init__(self, text_llm_infer: LLMInfer, sidebar: Sidebar, prompt: str, init_message="ask me anything"):
        colored_header(label=f"AI model", description="", color_name="gray-80")

        robot_avator = "🤖"
        user_avator = "🧑"

        if "model_history" not in st.session_state:
            st.session_state.model_history = {}
        if sidebar.role not in st.session_state.model_history:
            st.session_state.model_history[sidebar.role] = []

        if "messages" not in st.session_state:
            st.session_state.messages = {}
        if sidebar.role not in st.session_state.messages or len(st.session_state.messages[sidebar.role])==0:
            st.session_state.messages[sidebar.role] = [{"role": "robot", "content": init_message, "avatar": robot_avator}]

        # Display chat messages from history on app rerun
        for message in st.session_state.messages[sidebar.role]:
            with st.chat_message(message["role"], avatar=message.get("avatar")):
                st.markdown(message["content"])

        # Accept user input
        if message_input := st.chat_input("(Shift+Enter，Enter to send)："):
            # Display user message in chat message container
            with st.chat_message("user", avatar=user_avator):
                st.markdown(message_input)
            print('sss sss', st.session_state.messages[sidebar.role])
            model_request = combine_model_predict_request(st.session_state.messages[sidebar.role], message_input, prompt, sidebar.model_request)

            # Add user message to chat history
            st.session_state.messages[sidebar.role].append({"role": "user", "content": message_input, "avatar": user_avator})

            with st.chat_message("robot", avatar=robot_avator):
                message_placeholder = st.empty()
                for cur_response, model_history in generate_interactive(text_llm_infer=text_llm_infer, request=model_request):
                    st.session_state.model_history[sidebar.role] = model_history
                    # Display robot response in chat message container
                    message_placeholder.markdown(cur_response)
                # message_placeholder.markdown(cur_response)
            # Add robot response to chat history
            st.session_state.messages[sidebar.role].append({"role": "robot", "content": cur_response, "avatar": robot_avator})
        return
