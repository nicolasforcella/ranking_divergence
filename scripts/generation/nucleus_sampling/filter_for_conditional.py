# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
In order to avoid starting a generation in the middle of random text,
we start at the end of a paragraph, as long as the total length of text
to be conditioned on is less than 40 tokens. 40 is an arbitrary number
chosen because it is small enough that we could run a lot of experiments
with limited GPUs :)

Copied and adapted from https://github.com/ari-holtzman/degen/blob/master/filter_for_conditional.py
"""

import argparse
import json

from transformers import GPT2Tokenizer
from pathlib import Path
from ranking_divergence.data import DUO_OWT_CACHE_DIR, OWT_HELDOUT_SPLIT, load_openwebtext_texts

# For some reason HuggingFace only represent newlines inbetween non-whitespace tokens.
# So we hardcode this in to avoid strange, uninterpretable workarounds
NEWLINE = 198
HF_CACHE_DIR = Path("../.cache/huggingface/hub/datasets")

def sublist_end_index(list1, list2):
    for i in range(len(list2) - len(list1) + 1):
        if list2[i:i + len(list1)] == list1:
            return i + len(list1)
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", type=Path,
                        help="dir to write output to")
    parser.add_argument('--model-name', type=str, default='gpt2-large',
                        help='pretrained model name')
    parser.add_argument('--cache-dir', type=Path, default=HF_CACHE_DIR,
                        help='Directory to cache the OpenWebText dataset')
    parser.add_argument('-n', type=int, default=5000)
    parser.add_argument('-m', type=int, default=40)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print(args)

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name, do_lower_case=True)


    reference_texts = load_openwebtext_texts(
                split=OWT_HELDOUT_SPLIT,
                cache_dir=args.cache_dir,
                limit= 10 * args.n, # Add a buffer for filtering.
    )

    num = 0
    truncated_texts = []
    reference_texts_proccessed = []
    for text in reference_texts:
        tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        idx = sublist_end_index([NEWLINE, NEWLINE], tokenized)
        if idx is not None and idx < args.m:
            tokenized = tokenized[:idx]
            truncated_texts.append(tokenizer.decode(tokenized))
            reference_texts_proccessed.append(text)
            num += 1
            if num >= args.n:
                break
    if num < args.n:
        raise RuntimeError(f"Only could process {num} out of {args.n} prompts requested after filtering.")

    output_path = args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path / 'reference_texts.json', 'w') as output_file:
        output_file.write(json.dumps({'reference_texts': reference_texts_proccessed, 'truncated_texts': truncated_texts}))

    print(f"Wrote {num} reference texts to {output_path / 'reference_texts.json'}")
if __name__ == '__main__':
    main()