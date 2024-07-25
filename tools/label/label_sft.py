import asyncio
import json
import logging
import re
import time

import requests
from retrying import retry


class GPTClient:
    def __init__(
        self,
        token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmOGM3NjlmZWY1NTExMWVkOTJmMzAwMTYzZTA2MGU2YiIsImV4cCI6MTExNDUyNTk0OTh9.eY2VPBKyptzB897CguYRaxuY5aoqPrE8dqMhh7Dc0cM",
        url="https://openai.muxuekeji.cn/v1/chat/completions",
        model="gpt-3.5-turbo",
    ):
        self.url = url
        self.token = token  # self.get_token()
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {self.token}",
        }

    def get_token(self):
        token_post = requests.post(
            url="https://openai.muxuekeji.cn/v1/auth/token",
            headers={"Content-Type": "application/json"},
            json={"user_name": "evaluation", "expires": 1},
            timeout=60,
        )
        if token_post.status_code != 200:
            print("openai.muxuekeji token error")
        token = token_post.json()["access_token"]
        print(f"use token: {token}")
        return token

    # async def get_response(self, prompt):
    @retry(stop_max_attempt_number=99, wait_random_min=1000, wait_random_max=2000)
    def get_response(self, prompt):
        r = requests.post(
            self.url,
            headers=self.headers,
            json={
                "model": self.model,
                "messages": [
                    {"role": "user", "content": f"""{prompt}"""},
                ],
                "max_tokens": 1000,
                "temperature": 1,
                "top_p": 0.7,
                "stream": False,
            },
        )
        # print(r.json())
        res = r.json()
        # print(res)
        if "error" in res:
            print("retrying...", res)
            raise ValueError(res)
        return res["choices"][0]["message"]["content"]


# Now dialog begin: you only generate **one student question each turn**!,I will play the `<teacher>‘s` role and reply your question.**do not generate any <teacher>'s reply**!!,Output the student's question in one line with <student> </student> tag
#
def get_student_question_prompt(dialog):
    STUDENT_META_PROMPT = """
You are a student who has just finished the college entrance examination and you are seeking advice from me regarding university enrollment. You will ask me some questions related to enrollment, and at the same time, I will need you to provide some personal information related to your college entrance examination as a reference, such as your place of origin, your scores in the Chinese and math subjects, whether you took the arts or science stream, and your intended city of study, etc. These pieces of information can be provided by you when you ask me, for example:

<dialog> 
<student>: 老师你好，我是四川考生，我考了530分，想学计算机有什么推荐吗</student> 
<teacher>: 你好，请问你有意向的大学或者意向的城市吗？</teacher> 
</dialog>

Alternatively, they can be obtained by me from multiple conversations with you, for example:

<dialog> 
<student>: 我想咨询一下报考相关的问题</student> 
<teacher>: 你好，为了更好的回答您的问题。请问你可以提供一下你的高考相关信息吗？</teacher> 
<student>: 我是四川考生，考了420分</student> <teacher>: 好的，请问您是文科还是理科？</teacher> 
<student>: 文科</student> 
<teacher>: 请问您有意向的大学、意向专业或者意向城市吗？</teacher> 
</dialog>

For each of your questions, I will answer and guide you to provide me with some information related to your college entrance examination intentions. You will play the role of a college entrance examination student and ask me these questions. Please use your imagination to play this role well. 
notice all dialog need to in Chinese. 
you can only generate one student question each turn** and only play student role,Output the student's question in one line with <student> </student> tag.do not generate any <teacher>'s reply! and multi-turn dialog is forbidden!.again ! do not generate any <teacher>'s reply!!.Now dialog begin
<dialog>\n
"""
    return STUDENT_META_PROMPT + dialog


def get_teacher_reply_prompt(dialog):
    TEACHER_META_PROMPT = """
You are an educational consultant, your purpose is to provide assistance and guidance to students and parents in China based on something I have at my disposal (knowledge base resources, user profiles, or previous conversation records) related to education field.
you are currently conducting educational consultation with your student who has just finished the college entrance examination.You will answer student some questions related to enrollment, and at the same time, 
You need to guide student to provide some personal information related to student's college entrance examination as a reference, such as student's place of origin, scores,文理分科, and intended city of study, etc. These pieces of information can be provided by student when student ask you, for example:

<dialog> 
<student>: 老师你好，我是四川考生，我考了530分，想学计算机有什么推荐吗</student> 
<teacher>: 你好，请问你有意向的大学或者意向的城市吗？</teacher> 
</dialog>

If  student's question does not contain personal information related to the National College Entrance Examination, such as scores or province, you should first guide them to provide this information.for example:
<dialog> 
<student>: 我想咨询一下报考相关的问题</student> 
<teacher>: 你好，为了更好的回答您的问题。请问你可以提供一下你的高考相关信息吗？</teacher> 
<student>: 我是四川考生，考了420分</student> <teacher>: 好的，请问您是文科还是理科？</teacher> 
<student>: 文科</student> 
<teacher>: 请问您有意向的大学、意向专业或者意向城市吗？</teacher> 
</dialog>

For each of your questions, You will answer and guide student to provide you with some information related to student's college entrance examination intentions. You will play the role of <teacher> and I'll play the role of <student>. 
Please use your imagination to play this role well. notice all dialog need to be Chinese. **you only generate one student reply each turn** and only play teacher role,I will play the `<student>‘s` role and reply your question.**do not generate any <student>'s question!**.
notice: Output the teacher's reply in one line with <teacher> </teacher> tag,do not generate any <student>'s question! and generate multi-turn dialog is forbidden!,again. do not generate any <student>'s question! Now dialog begin:
<dialog>\n
"""
    return TEACHER_META_PROMPT + dialog


def get_user_profiler_prompt(dialog, user_profile):
    USER_PROFILER_META_PROMPT = f"""
You need to complete the user profile for Chinese Gaokao based on conversation history, historical user profile, and user input.

The default user profile has the following fields and datatypes:
性格:str, 年龄:int, 高考分数:int, 生源地:str, 科类:str, 心理状态:str,大学专业偏好:str,大学城市偏好:str,具体大学偏好:str,就业偏好:str,家庭情况:str
Note that "科类" will only have a value from [文科,理科,综合,物理,历史], feel free to leave this field empty unless the user specifically mentions which province they are from.

Please read the following information carefully and try to extract info for relevant fields in user profile in chinese.

<>
Conversation history:
<conversation>
{dialog}
</conversation>
Historical user profile: {user_profile}
</>

Output the updated profile in one line and in a clear, concise, and short manner.need to be Chinese. Do not explain or answer the question. Word limit is 300. only extract user profile info from <student>'s dialog.

The updated user profile:
"""
    return USER_PROFILER_META_PROMPT


# 提取回答
def extract_answer_from_dialog(dialog, type="student"):
    prefix = "<student>" if type == "student" else "<teacher>"
    suffix = "</student>" if type == "student" else "</teacher>"

    try:
        regex = f"{prefix}(.*){suffix}"
        matches = re.findall(regex, dialog)
        ast_dialog = matches[0].strip()
    except:
        ast_dialog = re.sub(f"{prefix}", "", dialog)
        ast_dialog = re.sub(f"{suffix}", "", ast_dialog)
        ast_dialog = re.sub(r"</dialog>", "", ast_dialog)
        ast_dialog = re.sub(r"<dialog>", "", ast_dialog)
        ast_dialog = re.sub(r":", "", ast_dialog)

    return ast_dialog


# 创建对话循环
async def gen_dialog(dialog_turn=3):
    name = f"coroutine-{time.time()}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(f"./logs/{name}.log")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    await asyncio.sleep(1)
    student = GPTClient()
    teacher = GPTClient()
    user_profiler = GPTClient()
    while True:
        conversation = ""
        history_profile = "首次对话，无历史画像"
        history_dialog_and_profile = {}

        for i in range(dialog_turn):
            # print(f"turn -------{i+1}--------")
            logger.debug(f"turn -------{i+1}--------")

            student_question = get_student_question_prompt(f"{conversation}")
            student_response = student.get_response(student_question)
            await asyncio.sleep(0.2)
            student_chat_data = extract_answer_from_dialog(student_response)

            # print(f"student: {student_chat_data}")
            logger.debug(f"student: {student_chat_data}")
            conversation += f"<student>: {student_chat_data}\n</student>\n"

            user_profiler_question = get_user_profiler_prompt(
                conversation, history_profile
            )
            user_profiler_response = user_profiler.get_response(user_profiler_question)
            await asyncio.sleep(0.2)

            # print(f"user_profiler: {user_profiler_response}")
            logger.debug(f"user_profiler: {user_profiler_response}")
            history_profile = user_profiler_response

            teacher_question = get_teacher_reply_prompt(f"{conversation}")
            teacher_response = teacher.get_response(teacher_question)
            await asyncio.sleep(0.2)
            teacher_chat_data = extract_answer_from_dialog(
                teacher_response, type="teacher"
            )
            # print(f"teacher: {teacher_chat_data}")
            logger.debug(f"teacher: {teacher_chat_data}")
            conversation += f"<teacher>: {teacher_chat_data}\n</teacher>\n"
            dialog_key = f"turn_{i+1}"
            turn_d = {}
            turn_d["student"] = student_chat_data
            turn_d["teacher"] = teacher_chat_data
            turn_d["user_profiler"] = history_profile
            history_dialog_and_profile[dialog_key] = turn_d
            await asyncio.sleep(0.2)

        # 将json写入文件
        time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        file_name = f"dialog_and_user_profile/dialog_{time_stamp}.json"
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(history_dialog_and_profile, f, ensure_ascii=False, indent=4)


async def main():
    asyncio.create_task(gen_dialog(3))
    asyncio.create_task(gen_dialog(3))
    asyncio.create_task(gen_dialog(3))
    asyncio.create_task(gen_dialog(3))
    asyncio.create_task(gen_dialog(3))
    asyncio.create_task(gen_dialog(3))
    asyncio.create_task(gen_dialog(3))
    asyncio.create_task(gen_dialog(3))
    asyncio.create_task(gen_dialog(3))
    asyncio.create_task(gen_dialog(3))

    # await asyncio.sleep(1000)
    while True:
        await asyncio.sleep(1000)


if __name__ == "__main__":
    asyncio.run(main())
