# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import argparse
import re
import logging
import transformers  # noqa: F401
import os
import json
from transformers import pipeline, set_seed
from transformers import AutoConfig, OPTForCausalLM, AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",
                        type=str,
                        help="Directory containing trained actor model")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate per response",
    )
    args = parser.parse_args()
    return args


def get_generator(path):
    if os.path.exists(path):
        # Locally tokenizer loading has some issue, so we need to force download
        model_json = os.path.join(path, "config.json")
        if os.path.exists(model_json):
            model_json_file = json.load(open(model_json))
            model_name = model_json_file["_name_or_path"]
            tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True,
                                                      fast_tokenizer=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, fast_tokenizer=True, trust_remote_code=True)

    tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True,
                                           from_tf=bool(".ckpt" in path),
                                           config=model_config).half()

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    #model.resize_token_embeddings(len(tokenizer))
    generator = pipeline("text-generation",
                         model=model,
                         tokenizer=tokenizer,
                         device="cuda:6")
    return generator


def get_user_input(user_input):
    tmp = input("Enter input (type 'quit' to exit, 'clear' to clean memory): ")
    new_inputs = f"Human: {tmp}\n Assistant: "
    user_input += f" {new_inputs}"
    return user_input, tmp == "quit", tmp == "clear"


def get_model_response(generator, user_input, max_new_tokens):
    response = generator(user_input, max_new_tokens=max_new_tokens)
    return response


def process_response(response, num_rounds):
    output = str(response[0]["generated_text"])
    output = output.replace("<|endoftext|></s>", "")
    all_positions = [m.start() for m in re.finditer("Human: ", output)]
    place_of_second_q = -1
    if len(all_positions) > num_rounds:
        place_of_second_q = all_positions[num_rounds]
    if place_of_second_q != -1:
        output = output[0:place_of_second_q]
    return output


def main(args):
    generator = get_generator(args.path)
    set_seed(42)

    user_input = ""
    num_rounds = 0
    while True:
        num_rounds += 1
        user_input, quit, clear = get_user_input(user_input)

        if quit:
            break
        if clear:
            user_input, num_rounds = "", 0
            continue

        response = get_model_response(generator, user_input,
                                      args.max_new_tokens)
        output = process_response(response, num_rounds)

        print("-" * 30 + f" Round {num_rounds} " + "-" * 30)
        print(f"{output}")
        user_input = f"{output}\n\n"


if __name__ == "__main__":
    # Silence warnings about `max_new_tokens` and `max_length` being set
    logging.getLogger("transformers").setLevel(logging.ERROR)

    args = parse_args()
    main(args)
