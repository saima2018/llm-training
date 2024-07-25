import argparse
import os

import dotenv
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(tensor_dict):
    for key, tensor in tensor_dict.items():
        tensor_dict[key] = tensor.to(device)


app = FastAPI()
model = None
tokenizer = None
dotenv.load_dotenv()


class LLMRequest(BaseModel):
    prompt: str = "hi there"
    max_new_tokens: int = 100
    repetition_penalty: float = 1.0
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 5


@app.post("/llm/inference")
def inference(llm_request: LLMRequest):
    global model, tokenizer
    text = llm_request.prompt
    # text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors="pt")
    print(f"prompt: {text}")
    to_device(encoded_input)
    output = model.generate(
        **encoded_input,
        temperature=llm_request.temperature,
        top_k=llm_request.top_k,
        top_p=llm_request.top_p,
        repetition_penalty=llm_request.repetition_penalty,
        max_length=llm_request.max_new_tokens,
        do_sample=True,
    )
    txt = tokenizer.decode(output[0], skip_special_tokens=False)
    return {"message": txt}


@app.on_event("startup")
async def startup_event():
    print("Starting up...")
    global model, tokenizer
    model_name_or_path = os.getenv("MODEL_NAME_OR_PATH", "gpt2")
    model_name_or_path = "./modeling/baichuan_half"
    print("model path", model_name_or_path)

    print(f"Loading model from {model_name_or_path}")
    if "gpt2" in model_name_or_path:
        model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(device)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_name_or_path)
    else:
        model = (
            AutoModelForCausalLM.from_pretrained(
                model_name_or_path, trust_remote_code=True
            )
            .half()
            .cuda()
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    reload = os.getenv("RELOAD", False)

    uvicorn.run("main:app", host=host, port=port, reload=False)
