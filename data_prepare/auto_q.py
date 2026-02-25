import json
import random
from openai import OpenAI
import base64
import re
from pandarallel import pandarallel
import pandas as pd
import os

pandarallel.initialize(nb_workers=64, progress_bar=True)

openai_api_key = "EMPTY"  
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


def post_messages(source_input):
    source_splited = [item for item in source_input.split('\n') if item]
    target_result = []

    for item in source_splited:
        pattern = r"^\d+\.\s*"
        cleaned_question = re.sub(pattern, "", item)
        cleaned_question = cleaned_question.strip()
        target_result.append(cleaned_question)
    return target_result


def init_message(info):
    messages = [{
        "role": "user",
        "content": [],
    }]
    if 'image' in info:
        messages[0]['content'].append({
            "type": "image_url",
            "image_url": {
                "url": process_image(info['image'])
            }
        })
    if 'text' in info:
        messages[0]['content'].append({"type": "text", "text": info['text']})
    messages[0]['content'].append({"type": "text", "text": '''You are an advanced AI assistant trained to analyze images and generate meaningful questions that capture their most important information. Analyze the given image and generate 3-5 specific questions that capture its most important visual information for retrieval purposes. Each question should:

Focus on distinct key elements (objects, actions, settings)
Be clear and answerable from visual content alone
Avoid subjective interpretations

Consider, but not limited to the following questions:

Main objects and their attributes (type, color, position)
Scene context (location type, time/weather if apparent)
Visible text/logos
Notable relationships between elements

Format your response as a numbered list with exactly one question per line.'''})
    return messages


def multi_generation(info):
    if '.jpg' not in info['image']:
        return 'Image is empty'
    init_messages = init_message(info)
    response = gpt_call(init_messages)
    return response


if __name__ == '__main__':
    image_files = '' ## source dir  ## format json
    save_dir = '' ## save dir ##format json
    files = [os.path.join(image_files, item) for item in os.listdir(image_files)]
    files.sort()
    for single in files:
        task_name = os.path.basename(single)
        save_path = os.path.join(save_dir, task_name)

        with open(single, mode='r', encoding='utf-8') as rf:
            data = json.load(rf)

        target_df = pd.DataFrame({'image': data})
        target_df['response'] = ['error'] * len(target_df)
        failed_df = target_df[target_df['response'] == 'error']
        success_df = target_df[target_df['response'] != 'error']

        while (len(failed_df) > 500):
            failed_df['response'] = failed_df.parallel_apply(multi_generation, axis=1)
            cur_success = failed_df[failed_df['response'] != 'error']
            success_df = pd.concat([success_df, cur_success], axis=0)
            success_df.to_json(save_path)
            failed_df = failed_df[failed_df['response'] == 'error']

