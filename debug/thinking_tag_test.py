# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from vllm.transformers_utils.tokenizer import get_tokenizer

tokenizer = get_tokenizer("Qwen/Qwen3-1.7B")

sys_message = {
    "role": "system",
    "content": "You are an expert BlackJack player. Output only 'HIT' or 'STAND'.",
}

user_message = {"role": "user", "content": "Hand: 15, Dealer: 10"}

assistant_message_partial = {"role": "assistant", "content": "<think>PARTIAL THINKING"}

messages = [
    sys_message,
    user_message,
    assistant_message_partial,
]

for add_generation_prompt in [True, False]:
    for tokenize in [True, False]:
        for enable_thinking in [True, False]:
            print(
                f"add_generation_prompt={add_generation_prompt}, "
                f"tokenize={tokenize}, "
                f"enable_thinking={enable_thinking}"
            )
            msg_with_chat_template = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
                enable_thinking=enable_thinking,
            )
            if tokenize:
                print(
                    f"msg_with_chat_template decoded: {tokenizer.decode(msg_with_chat_template)}"
                )
            else:
                print(f"msg_with_chat_template: {msg_with_chat_template}")
            print("=" * 5)


print("NOW COMPLETE THINKING")

assistant_message_complete = {
    "role": "assistant",
    "content": "<think>COMPLETE THINKING</think>",
}
messages = [
    sys_message,
    user_message,
    assistant_message_complete,
]

for add_generation_prompt in [True]:
    for tokenize in [True]:
        for enable_thinking in [True, False]:
            print(
                f"add_generation_prompt={add_generation_prompt}, "
                f"tokenize={tokenize}, "
                f"enable_thinking={enable_thinking}"
            )
            msg_with_chat_template = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
                enable_thinking=enable_thinking,
            )

            if tokenize:
                print(
                    f"msg_with_chat_template decoded: {tokenizer.decode(msg_with_chat_template)}"
                )
            else:
                print(f"msg_with_chat_template: {msg_with_chat_template}")
            print("=" * 5)

print("NO THINKING")
assistant_message_no_thinking = {
    "role": "assistant",
    "content": "NO THINKING CONTENT",
}
messages = [
    sys_message,
    user_message,
    assistant_message_no_thinking,
]

for add_generation_prompt in [True]:
    for tokenize in [True]:
        for enable_thinking in [True, False]:
            print(
                f"add_generation_prompt={add_generation_prompt}, "
                f"tokenize={tokenize}, "
                f"enable_thinking={enable_thinking}"
            )
            msg_with_chat_template = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
                enable_thinking=enable_thinking,
            )

            if tokenize:
                print(
                    f"msg_with_chat_template decoded: {tokenizer.decode(msg_with_chat_template)}"
                )
            else:
                print(f"msg_with_chat_template: {msg_with_chat_template}")
            print("=" * 5)
