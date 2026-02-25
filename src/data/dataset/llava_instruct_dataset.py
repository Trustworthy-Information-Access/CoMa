import os
import re

import datasets
from src.data.dataset.base_qa_dataset import AutoQADataset, add_metainfo_hook, MULTIMODAL_FEATURES, \
    RESOLUTION_MAPPING
from src.model.processor import VLM_IMAGE_TOKENS, VLM_COMPRESS_TOKENS
from src.utils import print_master, print_rank

def process_multiturn_conversations(conversations, image_token, compress_tokens, prompt, compression={}):
    if len(compress_tokens) > 0:
        encode_query = image_token * len(re.findall(r"<image>", conversations[0]["value"]))
        if compression.get('query', None) is not None:
            encode_query = encode_query + '\n' + compression['query']
        encode_answer = "".join(compress_tokens)
        conversations[0]["value"] = conversations[0]["value"].replace("<image>", "")
        decode_query, decode_answer = [], []
        for i in range(0, len(conversations), 2):
            assert conversations[i]['from'] == 'human'
            assert conversations[i + 1]['from'] == 'gpt'
            decode_query.append(conversations[i]["value"])
            decode_answer.append(conversations[i + 1]["value"])
        if prompt:
            encode_query = encode_query + prompt
    else:   # vanilla llava instruct dataset
        raise ValueError("No supported for no compress_tokens")
    return encode_query, encode_answer, decode_query, decode_answer

def process_conversations(conversations, image_token, compress_tokens, prompt, compression={}):
    if len(compress_tokens) > 0:
        encode_query = image_token * len(re.findall(r"<image>", conversations[0]["value"]))
        if compression.get('query', None) is not None:
            encode_query = encode_query + '\n' + compression['query']
        encode_answer = "".join(compress_tokens)
        decode_query = conversations[0]["value"].replace("<image>", "")
        decode_answer = conversations[1]["value"]
        if prompt:
            encode_query = encode_query + prompt
    else:   # vanilla llava instruct dataset
        encode_query = None
        encode_answer = None
        decode_query = conversations[0]["value"].replace("<image>", image_token)
        decode_answer = conversations[1]["value"]
    return encode_query, encode_answer, decode_query, decode_answer

def process_raw_conversations(conversations, image_path):
    processed_conversation = []
    for i in range(0, len(conversations), 2):
        assert conversations[i]['from'] == 'human'
        assert conversations[i + 1]['from'] == 'gpt'
        processed_conversation.extend([
            {
                "role": "user",
                "content": [{"type": "text", "text": conversations[i]['value'].replace('<image>', '')}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": conversations[i + 1]['value']}],
            }
        ])
    if image_path:
        processed_conversation[0]["content"] = [{"type": "image", "image": image_path}] + processed_conversation[0]["content"]
    return str(processed_conversation)

@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    model_backbone = kwargs['model_backbone']
    image_resolution = kwargs['image_resolution']
    image_dir = kwargs['image_dir']
    multi_turn = kwargs.get('multi_turn', False)
    batch_size = len(batch_dict['id'])
    import json
    encode_queries, encode_answers, encode_images, decode_queries, decode_answers = [], [], [], [], []
    for data_idx, (data_id, conversations, image_path, compression) in enumerate(zip(batch_dict['id'], batch_dict['conversations'], batch_dict['image'], batch_dict.get('compression', [json.dumps({'image': None, 'query': None}) for i in range(batch_size)]))):
        compression = json.loads(compression)
        try:
            if image_path is None and compression.get('image', None) is None:
                encode_prompt = "\nSummarize above text in a few words."
            else:
                encode_prompt = "\nSummarize above image and text in a few words."
            if multi_turn and len(conversations) > 2:
                encode_query, encode_answer, decode_query, decode_answer = process_multiturn_conversations(
                    conversations, 
                    image_token=VLM_IMAGE_TOKENS[model_backbone], 
                    compress_tokens=VLM_COMPRESS_TOKENS, 
                    prompt=encode_prompt,
                    compression=compression,
                )
            else:
                encode_query, encode_answer, decode_query, decode_answer = process_conversations(
                    conversations, 
                    image_token=VLM_IMAGE_TOKENS[model_backbone], 
                    compress_tokens=VLM_COMPRESS_TOKENS, 
                    prompt=encode_prompt,
                    compression=compression,
                )
                decode_query = [decode_query]
                decode_answer = [decode_answer]
            encode_image = {"bytes": [None], "paths": [os.path.join(image_dir, image_path) if image_path else None], "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]}
            encode_queries.append(encode_query)
            encode_answers.append(encode_answer)
            encode_images.append(encode_image)
            decode_queries.append(decode_query)
            decode_answers.append(decode_answer)
        except Exception as e:
            print(f'Error in processing {DATASET_PARSER_NAME}: \n\t\tdata id: {data_id} \n\t\tconversations: {conversations}')
            print(e)
            raise e

    raw_conversations = [
        process_raw_conversations(conv, os.path.join(image_dir, image_path) if image_path else None)
        for conv, image_path in zip(batch_dict['conversations'], batch_dict['image'])
    ]
    return {"encode_query": encode_queries, "encode_answer": encode_answers, "encode_image": encode_images,
            "decode_query": decode_queries, "decode_answer": decode_answers, "raw_conversations": raw_conversations}

DATASET_PARSER_NAME = "llava_instruct"
@AutoQADataset.register(DATASET_PARSER_NAME)
def load_llava_instruct_dataset(model_args, data_args, training_args, *args, **kwargs):
    dataset_name = kwargs.get("dataset_name", DATASET_PARSER_NAME)
    assert "dataset_path" in kwargs, "`dataset_path` should be given for loading llava instruct dataset."
    assert "image_dir" in kwargs, "`image_dir` should be given for loading llava instruct dataset."
    dataset_path = kwargs["dataset_path"]
    dataset = datasets.load_dataset("json", split="train", data_files=dataset_path, streaming=False)
    dataset = dataset.shuffle(seed=training_args.seed)
    num_sample_per_subset = kwargs.get("num_sample_per_subset", getattr(data_args, "num_sample_per_subset", None))
    if num_sample_per_subset is not None and num_sample_per_subset < dataset.num_rows:
        num_rows = int(num_sample_per_subset)
        dataset = dataset.select(range(num_rows))

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution
    kwargs['image_dir'] = kwargs["image_dir"]
    kwargs['multi_turn'] = training_args.multi_turn
    kwargs['global_dataset_name'] = f'{DATASET_PARSER_NAME}/{dataset_name}'
    dataset = dataset.shuffle(seed=training_args.seed)
    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True, batch_size=128, drop_last_batch=False)
    dataset = dataset.select_columns(["encode_query", "encode_answer", "encode_image", "decode_query", "decode_answer", "global_dataset_name", "task_type", "raw_conversations"])
    # dataset = dataset.cast(MULTIMODAL_FEATURES)
    print_master(f"Loaded {DATASET_PARSER_NAME}/{dataset_name} dataset with {len(dataset)} samples")

    return dataset