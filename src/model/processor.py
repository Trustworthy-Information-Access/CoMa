import os
import logging

import PIL
from transformers.image_utils import ChannelDimension

logger = logging.getLogger(__name__)

import torch
import numpy as np
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from src.utils import print_master
from src.model.utils import get_rope_index_2, get_rope_index_25, pad_and_cat, shift_sep_indices

IGNORE_INDEX = -100

QWEN2_5_VL = 'qwen2_5_vl'
QWEN2_5_VL_COMPRESSION = 'qwen2_5_vl_compression'
MODEL2BACKBONE = {  # keys are from hf_config.model_type or manually added if not provided
    'qwen2_5_vl': QWEN2_5_VL,
    'qwen2_5_vl_compression': QWEN2_5_VL_COMPRESSION,
}
SUPPORTED_MODELS = set(MODEL2BACKBONE.keys())

VLM_IMAGE_TOKENS = {
    QWEN2_5_VL: "<|vision_start|><|image_pad|><|vision_end|>",
    QWEN2_5_VL_COMPRESSION: "<|vision_start|><|image_pad|><|vision_end|>",  # add vision start and end token for mrope
}

TOTAL_COMPRESS_TOKENS = int(os.environ.get("TOTAL_COMPRESS_TOKENS", 32))
VLM_COMPRESS_TOKENS = [f"<|ctoken_pad_{i}|>" for i in range(TOTAL_COMPRESS_TOKENS)]

VLM_VIDEO_TOKENS = {
    QWEN2_5_VL: "<|video_pad|>",
    QWEN2_5_VL_COMPRESSION: "<|video_pad|>",
}

backbone2model = {}
from src.model.vlm_backbone.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
backbone2model[QWEN2_5_VL] = Qwen2_5_VLForConditionalGeneration

from src.model.vlm_backbone.qwen2_5_vl_compression import Qwen2_5_VLForConditionalGeneration as Qwen2_5_VL_CompressionForConditionalGeneration
backbone2model[QWEN2_5_VL_COMPRESSION] = Qwen2_5_VL_CompressionForConditionalGeneration

def load_processor(model_args, data_args=None):
    """
    Load processor based on VLM backbone.
    Note: due to this change, https://github.com/huggingface/transformers/commit/9215cc62d4366072aacafa4e44028c1ca187167b#diff-6505546ec5a9ab74b2ce6511681dd31194eb91e9fa3ce26282e487a5e61f9356L1102
    """
    model_name_or_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
    print_master(f'Loading processor from: {model_name_or_path}')
    if model_args.model_backbone == QWEN2_5_VL:
        from src.model.vlm_backbone.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
        from src.model.vlm_backbone.qwen2_5_vl.image_processing_qwen2_5_vl import Qwen2_5_VLImageProcessor
        from src.model.vlm_backbone.qwen2_vl.tokenization_qwen2_fast import Qwen2TokenizerFast
        min_pixels, max_pixels = None, None
        if data_args is not None:
            min_pixels, max_pixels = data_args.resize_min_pixels, data_args.resize_max_pixels
        size = {"shortest_edge": min_pixels, "longest_edge": max_pixels, "min_pixels": min_pixels, "max_pixels": max_pixels}
        image_processor = Qwen2_5_VLImageProcessor.from_pretrained(model_name_or_path, size=size)
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path)
        tokenizer.padding_side = "left"
        processor = Qwen2_5_VLProcessor.from_pretrained(model_name_or_path, image_processor=image_processor, tokenizer=tokenizer)
    elif model_args.model_backbone == QWEN2_5_VL_COMPRESSION:
        print_master("Using Qwen2_5_VL_COMPRESSION")
        from src.model.vlm_backbone.qwen2_5_vl_compression.processing_qwen2_5_vl import Qwen2_5_VLProcessor
        from src.model.vlm_backbone.qwen2_5_vl_compression.image_processing_qwen2_5_vl import Qwen2_5_VLImageProcessor
        from src.model.vlm_backbone.qwen2_vl.tokenization_qwen2_fast import Qwen2TokenizerFast
        min_pixels, max_pixels = None, None
        if data_args is not None:
            min_pixels, max_pixels = data_args.resize_min_pixels, data_args.resize_max_pixels
        size = {"shortest_edge": min_pixels, "longest_edge": max_pixels, "min_pixels": min_pixels, "max_pixels": max_pixels}
        image_processor = Qwen2_5_VLImageProcessor.from_pretrained(model_name_or_path, size=size)
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path)
        tokenizer.add_tokens(VLM_COMPRESS_TOKENS, special_tokens=True)  # add special tokens for compression
        processor = Qwen2_5_VLProcessor.from_pretrained(model_name_or_path, image_processor=image_processor, tokenizer=tokenizer)
        # remove 'system' in chat template
        processor.chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        # processor.tokenizer.padding_side = "left"
    else:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True,
        )
    return processor


def get_backbone_name(hf_config, model_type=None):
    if model_type is not None:
        setattr(hf_config, 'model_type', model_type)
    assert hf_config.model_type in SUPPORTED_MODELS, f"Unknown backbone name {hf_config.model_type}.Supported models are {SUPPORTED_MODELS}"
    return MODEL2BACKBONE[hf_config.model_type]

def Qwen2_VL_process_fn(model_inputs: dict, processor: "Qwen2VLProcessor", max_length=None):
    # TODO: set separate max_len for text/visual inputs, currently max_length is only applied to text-only data
    input_ids, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw = [], [], [], [], []
    position_ids = []
    texts, visual_inputs = model_inputs['text'], model_inputs['images']
    image_exists = False
    vlm_image_token, vlm_video_token = VLM_IMAGE_TOKENS[QWEN2_5_VL], VLM_VIDEO_TOKENS[QWEN2_5_VL]

    # 1. iterate each pair and process, since processors do not support processing for mixed batch (contains data w/ and w/o visual inputs)
    for text, images in zip(texts, visual_inputs):
        if images is None or (type(images)==list and any(i is None for i in images)):
            # all images must be valid
            inputs = processor(text=[text], images=None, return_tensors="pt", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_grid_thw.append(None)
            pixel_values_videos.append(None)
            video_grid_thw.append(None)
            position_ids.append(
                torch.arange(0, inputs["input_ids"].size(1))
                .view(1, -1)
                .unsqueeze(0)
                .expand(3, -1, -1)
            )
        else:
            try:
                if vlm_image_token in text:
                    if isinstance(images, PIL.Image.Image):
                        # images is a single image
                        images = [images]
                    for iid, image in enumerate(images):
                        # rare case in MMEB eval: resize to 28*28 if either w or h is smaller than 28
                        if image.size[0] < 28 or image.size[1] < 28:
                            image = image.resize((56, 56))
                            images[iid] = image
                    inputs = processor(text=[text], images=images, return_tensors="pt", max_length=max_length, truncation=False, input_data_format=ChannelDimension.LAST)
                elif vlm_video_token in text:
                    # TODO: check text/video data validity
                    inputs = processor(text=[text], videos=[images], return_tensors="pt", max_length=max_length, truncation=False, input_data_format=ChannelDimension.LAST)
                else:
                    raise NotImplementedError(f"No visual token found ({vlm_image_token} or {vlm_video_token}) in the text: {text}")
            except Exception as e:
                for i in images:
                    print(i.filename)
                raise e
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            if 'pixel_values' in inputs:
                pixel_values.append(inputs['pixel_values'])
                image_grid_thw.append(inputs['image_grid_thw'])
                pixel_values_videos.append(None)
                video_grid_thw.append(None)
            else:
                pixel_values.append(None)
                image_grid_thw.append(None)
                pixel_values_videos.append(inputs['pixel_values_videos'])
                video_grid_thw.append(inputs['video_grid_thw'])

            # rope position ids
            pos_ids, _ = get_rope_index_25(
                processor.image_processor.merge_size,
                inputs["input_ids"],
                image_grid_thw=inputs.get("image_grid_thw", None),
                video_grid_thw=inputs.get("video_grid_thw", None),
                second_per_grid_ts=inputs.get("second_per_grid_ts", None),
            )
            position_ids.append(pos_ids)

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    position_ids = pad_and_cat(position_ids, max_length=max_length)
    # manually enforce long type due to:
    # (1) [rank7]: RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)
    # (2) [rank7]:   File "/fsx/home/ruimeng/project/VLM2Vec/src/model.py", line 45, in _pooling
    #     [rank7]:     reps = last_hidden_state[
    #     [rank7]: IndexError: tensors used as indices must be long, int, byte or bool tensors
    inputs = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.long(), 
        'position_ids': position_ids.long(), 
        'texts': texts,
        'images': visual_inputs,
    }
    # if input_ids.shape[1] > 8192:
    #     if torch.distributed.get_rank() == 0:
    #         breakpoint()
    #     torch.distributed.barrier()
    inputs['pixel_values'] = pixel_values
    inputs['image_grid_thw'] = image_grid_thw
    inputs['pixel_values_videos'] = pixel_values_videos
    inputs['video_grid_thw'] = video_grid_thw

    return inputs

def Qwen2_VL_Compression_process_fn(model_inputs: dict, processor: "Qwen2VLProcessor", ref_processor=None, max_length=None, task_type="retrieval"):
    input_ids, position_ids, labels, sep_indices, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw = [], [], [], [], [], [], [], []
    texts, visual_inputs = model_inputs['text'], model_inputs['images']
    image_exists = False
    vlm_image_token, vlm_video_token = VLM_IMAGE_TOKENS[QWEN2_5_VL], VLM_VIDEO_TOKENS[QWEN2_5_VL]

    def calc_multiturn_label(target, sep_token_indices, start=0):
        # multi-turn 
        # mask all Q
        #                           0                                              1                         2                                            3
        # <Q1> <|im_end|> <\n> <|im_start|> <assistant> <\n> <A1> <|im_end|> <\n> <Q2> <|im_end|> <\n> <|im_start|> <assistant> <\n> <A2> <|im_end|> <\n>
        #  |                        |                         |                    |
        # prev                     cur                     cur + 3                prev                 
        prev = start
        label = target.clone().detach()
        for i in range(0, len(sep_token_indices), 2):
            curr = sep_token_indices[i]
            label[prev : curr + 3] = IGNORE_INDEX
            if i + 1 < len(sep_token_indices):
                # in case the "last" answer is truncated
                prev = sep_token_indices[i + 1]
        return label

    # 1. iterate each pair and process, since processors do not support processing for mixed batch (contains data w/ and w/o visual inputs)
    for text, images in zip(texts, visual_inputs):
        if images is None or (type(images)==list and any(i is None for i in images)):
            # assert task_type == "retrieval"
            # all images must be valid
            inputs = processor(text=[text], images=None, return_tensors="pt", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_grid_thw.append(None)
            pixel_values_videos.append(None)
            video_grid_thw.append(None)
            position_ids.append(
                torch.arange(0, inputs["input_ids"].size(1))
                .view(1, -1)
                .unsqueeze(0)
                .expand(3, -1, -1)
            )
        else:
            try:
                if vlm_image_token in text:
                    if isinstance(images, PIL.Image.Image):
                        # images is a single image
                        images = [images]
                    for iid, image in enumerate(images):
                        # rare case in MMEB eval: resize to 28*28 if either w or h is smaller than 28
                        if image.size[0] < 28 or image.size[1] < 28:
                            # TODO: 
                            image = image.resize((56, 56))
                            images[iid] = image
                    if task_type == "openqa":
                        inputs = processor(text=[text], images=images, return_tensors="pt", max_length=max_length, truncation=True, input_data_format=ChannelDimension.LAST)
                    elif task_type == "retrieval":
                        inputs = processor(text=[text], images=images, return_tensors="pt", max_length=max_length, truncation=False, input_data_format=ChannelDimension.LAST)
                elif vlm_video_token in text:
                    # TODO: check text/video data validity
                    if task_type == "openqa":
                        inputs = processor(text=[text], videos=[images], return_tensors="pt", max_length=max_length, truncation=True, input_data_format=ChannelDimension.LAST)
                    elif task_type == "retrieval":
                        inputs = processor(text=[text], videos=[images], return_tensors="pt", max_length=max_length, truncation=False, input_data_format=ChannelDimension.LAST)
                else:
                    raise NotImplementedError(f"No visual token found ({vlm_image_token} or {vlm_video_token}) in the text: {text}")
            except Exception as e:
                breakpoint()
                raise e
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            if 'pixel_values' in inputs:
                pixel_values.append(inputs['pixel_values'])
                image_grid_thw.append(inputs['image_grid_thw'])
                pixel_values_videos.append(None)
                video_grid_thw.append(None)
            else:
                pixel_values.append(None)
                image_grid_thw.append(None)
                pixel_values_videos.append(inputs['pixel_values_videos'])
                video_grid_thw.append(inputs['video_grid_thw'])

            # rope position ids
            pos_ids, _ = get_rope_index_25(
                processor.image_processor.merge_size,
                inputs["input_ids"],
                image_grid_thw=inputs.get("image_grid_thw", None),
                video_grid_thw=inputs.get("video_grid_thw", None),
                second_per_grid_ts=inputs.get("second_per_grid_ts", None),
            )
            position_ids.append(pos_ids)

        # labels & seq_indices
        if task_type == "openqa":
            target = inputs["input_ids"].squeeze()
            eos_token_id = processor.tokenizer.eos_token_id  # split by eos token
            compression_start_id = processor.tokenizer.encode(VLM_COMPRESS_TOKENS[0])[0]
            # (I,   C,   Q,   A)
            #   |    |    |
            #  sep1 sep2 sep3
            sep1_token_indices = np.where(target == compression_start_id)[0]
            assert len(sep1_token_indices) == 1, f"Expected compression tokens to be appended at the end of the input sequence. text: {text}"
            sep2_token_indices = sep1_token_indices + TOTAL_COMPRESS_TOKENS
            sep_token_indices = np.where(target == eos_token_id)[0] + 2  # <|im_end|>\n

            sep_token_indices = np.concatenate([sep1_token_indices, sep2_token_indices, sep_token_indices[:2]])
            assert target[sep_token_indices[0]] == processor.tokenizer.encode(VLM_COMPRESS_TOKENS[0])[0], f"Expected {VLM_COMPRESS_TOKENS[0]}, but got {target[sep_token_indices[0]]}"
            
            target[:sep_token_indices[-3]] = IGNORE_INDEX

            labels.append(torch.tensor(target, dtype=torch.long))
            sep_indices.append(sep_token_indices)
        elif task_type == "retrieval":
            compression_start_id = processor.tokenizer.encode(VLM_COMPRESS_TOKENS[0])[0]
            sep1_token_indices = np.where(inputs["input_ids"].squeeze() == compression_start_id)[0]
            sep2_token_indices = sep1_token_indices + TOTAL_COMPRESS_TOKENS
            assert sep2_token_indices[0] == len(inputs["input_ids"][0]) or inputs["input_ids"][0][sep2_token_indices[0]] == processor.tokenizer.eos_token_id, inputs["input_ids"][0][sep2_token_indices[0]]
            sep_indices.append(sep1_token_indices.tolist() + sep2_token_indices.tolist() + [len(input_ids[-1]), 0, 0])
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    position_ids = pad_and_cat(position_ids)
    sep_indices = torch.tensor(sep_indices, dtype=torch.long)
    assert sep_indices.shape[0] == input_ids.shape[0]
    # manually enforce long type due to:
    # (1) [rank7]: RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)
    # (2) [rank7]:   File "/fsx/home/ruimeng/project/VLM2Vec/src/model.py", line 45, in _pooling
    #     [rank7]:     reps = last_hidden_state[
    #     [rank7]: IndexError: tensors used as indices must be long, int, byte or bool tensors
    inputs = {
        'input_ids': input_ids.long(),
        'position_ids': position_ids.long(),
        'attention_mask': attention_mask.long(),
        'sep_indices': sep_indices,
        'images': visual_inputs,
    }

    if len(labels) > 0:
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        inputs['labels'] = labels.long()

    inputs['pixel_values'] = pixel_values
    inputs['image_grid_thw'] = image_grid_thw
    inputs['pixel_values_videos'] = pixel_values_videos
    inputs['video_grid_thw'] = video_grid_thw

    if 'ref_texts' in model_inputs:
        ref_inputs = processor(
            text=model_inputs['ref_texts'], 
            images=visual_inputs, 
            return_tensors="pt", 
            truncation=True, 
            input_data_format=ChannelDimension.LAST, 
            padding=True,
        )
        
        labels = []
        for input_ids in ref_inputs['input_ids']:
            sep_token_indices = np.where(input_ids == ref_processor.tokenizer.eos_token_id)[0] + 2
            label = calc_multiturn_label(input_ids, sep_token_indices)

            labels.append(label)
        ref_inputs['labels'] = torch.stack(labels)

        for k, v in ref_inputs.items():
            inputs[f'ref_{k}'] = v

    return inputs

def process_input_text(instruction, model_backbone, text=None, add_video_token=False, add_image_token=False):
    # Formulate input text based on text, special token and instruction.
    prompt = instruction
    if text:
        prompt = prompt + " " + text
    if add_video_token:
        video_token = VLM_VIDEO_TOKENS[model_backbone]
        prompt = video_token + " " + prompt
    if add_image_token:
        image_token = VLM_IMAGE_TOKENS[model_backbone]
        prompt = image_token + " " + prompt
    return prompt


def postprocess_input_text(query_texts, cand_texts, model_backbone):
    summarize_prompt = "".join(VLM_COMPRESS_TOKENS)
    query_texts = [[query + summarize_prompt for query in queries] for queries in query_texts ]
    cand_texts = [[cand + summarize_prompt for cand in cands] for cands in cand_texts]
    return query_texts, cand_texts


process_vlm_inputs_fns = {
    QWEN2_5_VL: Qwen2_VL_process_fn,
    QWEN2_5_VL_COMPRESSION: {
        "openqa": partial(Qwen2_VL_Compression_process_fn, task_type="openqa"), 
        "retrieval": partial(Qwen2_VL_Compression_process_fn, task_type="retrieval")
    },
}