import json
import random
from openai import OpenAI
import base64
import re
import copy
from tqdm import tqdm
import os
import pandas as pd
from pandarallel import pandarallel

pandarallel.initialize(nb_workers=64, progress_bar=True)

openai_api_key = "EMPTY"  # 填 EMPTY
openai_api_base = "http://localhost:10000/v1"  
model_name = '../Qwen2.5-VL-7B-Instruct'
client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)


def process_image(image_path):
    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read())
    encoded_image_text = encoded_image.decode("utf-8")
    base64_qwen = f"data:image;base64,{encoded_image_text}"
    return base64_qwen


def gpt_call(messages):
    # print('requset_message',messages)
    try:
        llm_response = client.chat.completions.create(
            messages=messages,
            model=model_name,  
            max_tokens=4096,
            temperature=0.7,
            stream=False  
        )
        response = llm_response.choices[0].message.content
    except Exception as e:
        print(e)
        response = 'error'
    return response


def init_message(info):
    messages = [{
        "role": "user",
        "content": [],
    }]
    if 'image' in info and info['image'] is not None and isinstance(info['image'], str):
        messages[0]['content'].append({
            "type": "image_url",
            "image_url": {
                "url": process_image(info['image'])
                # 'url':'None'
            }
        })
    return messages


def clean_string(s):
    
    cleaned = re.sub(r'^[^a-zA-Z]+', '', s)
    return cleaned


def construct_questions(row):
    questions = [item for item in row.split('\n') if len(item)]
    processed_row = []
    for single_q in questions:
        processed_row.append(clean_string(single_q))
    return processed_row


def request_single(row):
    questions = row['questions']
    image = row['image']
    single_record = {'message': []}
    single_record['image'] = image

    message = init_message({'image': image})
    questions = construct_questions(questions)
    for index, question in enumerate(questions):
        if index == 0:
            message[0]['content'].append({"type": "text", 'text': question})
            response = gpt_call(message)
            message.append({'role': 'assistant', 'content': response})
            single_record['message'].append({'from': 'human', 'content': question})
            single_record['message'].append({'from': 'gpt', 'content': response})
        else:
            message.append({'role': 'user', 'content': question})
            response = gpt_call(message)
            message.append({'role': 'assistant', 'content': response})
            single_record['message'].append({'from': 'human', 'content': question})
            single_record['message'].append({'from': 'gpt', 'content': response})
    return json.dumps(single_record)


if __name__ == '__main__':

    source_dir = '' ## auto_q
    save_dir = '' ## auto_a

    original_files = [item for item in os.listdir(source_dir) if '.json' in item]
    cache_files = [item for item in os.listdir(save_dir) if '.json' in item]
    unparsed_files = list(set(original_files) - set(cache_files))
    unparsed_files = sorted(unparsed_files)

    files = [os.path.join(source_dir, item) for item in unparsed_files]
    for single_file in files:

        source_df = pd.read_json(single_file)
        source_df = source_df[source_df['response'] != 'error']
        source_df = source_df[source_df['response'] != 'Image is empty']
        source_df.rename(columns={'response': 'questions'}, inplace=True)
        source_df['response'] = ['error'] * len(source_df)
        task_name = os.path.basename(single_file)
        failed_df = source_df[source_df['response'] == 'error']
        success_df = source_df[source_df['response'] != 'error']

        while (len(failed_df) > 500):
            failed_df['response'] = failed_df.parallel_apply(request_single, axis=1)
            cur_success = failed_df[failed_df['response'] != 'error']
            success_df = pd.concat([success_df, cur_success], axis=0)
            success_df.to_json(os.path.join(save_dir, task_name))
            failed_df = failed_df[failed_df['response'] == 'error']
