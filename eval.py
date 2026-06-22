# Reference: https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety

from tqdm import tqdm
import time
import sys
import json
import numpy as np
from openai import OpenAI
import argparse
import os
import eval_template

OPENAI_API_KEY = "YOUR API KEY"

def result_print(scores_attacked):
    filtered_scores = [score for score in scores_attacked[0] if isinstance(score, int)]
    scores_attacked_np = np.array(filtered_scores)
    scores_attacked_avg = scores_attacked_np.mean()
    return scores_attacked_avg

def extract_content(tag, text):
    start_idx = text.find(tag)
    if start_idx == -1:
        return None

    content_after_tag = text[start_idx + len(tag):].strip()
    parts = content_after_tag.split()
    if tag == "#thescore:":
        assert parts[0].isdigit()
        return int(parts[0])
    else:
        end_idx = text.find("#", start_idx + 1)
        return content_after_tag if end_idx == -1 else content_after_tag[:end_idx].strip()

def gpt_judge(QApairs, outputs_to_judge):
    responses = []
    for QApair in tqdm(QApairs):
        while True:
            try:
                client = OpenAI(api_key=OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model= "gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "user",
                            "content": eval_template.score_template % QApair
                        }
                    ],
                    temperature=0,
                    max_tokens=4096,
                    top_p=0,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                responses.append(response)
                break
            except Exception as err:
                break

    contents = [response.choices[0].message.content for response in responses]
    scores = [extract_content("#thescore:", content) for content in contents]
    reasons = [extract_content("#thereason:", content) for content in contents]
    return scores, reasons, outputs_to_judge

def ahs(file_path):
    data_by_no = {}
    data_without_no = []
    with open(file_path) as f:
        lines = f.readlines()

    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        if "Input" not in record or "Response" not in record:
            continue
        sample_no = record.get("No.")
        if isinstance(sample_no, bool):
            sample_no = None
        if isinstance(sample_no, int):
            data_by_no[sample_no] = record
        else:
            data_without_no.append(record)

    if data_by_no:
        data = [data_by_no[key] for key in sorted(data_by_no)]
    else:
        data = data_without_no
    if not data:
        return None

    prompt = [data[i]['Input'] for i in range(len(data))]
    outputs_to_judge = [data[i]['Response'] for i in range(len(data))]
    QApairs = list(zip(prompt, outputs_to_judge))

    duo_results = gpt_judge(QApairs, outputs_to_judge)
    scores_attacked_avg = result_print(duo_results)

    return scores_attacked_avg
