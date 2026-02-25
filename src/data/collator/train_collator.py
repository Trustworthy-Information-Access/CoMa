from itertools import repeat
from typing import Any, Optional
from torch.jit import isinstance

import logging
from dataclasses import dataclass
from transformers import ProcessorMixin, AutoProcessor, AutoTokenizer
from src.arguments import DataArguments, ModelArguments, TrainingArguments
import torch
from qwen_vl_utils import smart_resize

from src.model.processor import QWEN2_5_VL, QWEN2_5_VL_COMPRESSION, process_vlm_inputs_fns
from PIL import Image
import io
from src.utils import print_rank, print_master


logger = logging.getLogger(__name__)

def split_and_process_vlm_inputs(model_input: dict, chunk_size: int):
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]

    keys = list(arg_val.keys())
    vals = [arg_val[k] for k in keys]
    bss = [len(arg_val[k]) if k != 'position_ids' else arg_val[k].shape[1] for k in keys if arg_val[k] is not None]
    assert len(bss) > 0 and all(bs == bss[0] for bs in bss), list(zip(keys, bss, [val[0] if not isinstance(val, torch.Tensor) else val.shape for val in vals]))
    bs = bss[0]
    chunked_tensors = []
    for k in keys:
        if arg_val[k] is None:
            chunked_tensor = [None for _ in range(bs)]
        elif isinstance(arg_val[k], torch.Tensor):
            # for position_ids, we need to split along dim=1
            chunked_tensor = arg_val[k].split(chunk_size, dim=0) if k != "position_ids" else arg_val[k].split(chunk_size, dim=1)
        else:
            chunked_tensor = [arg_val[k][i: i + chunk_size] for i in list(range(0, len(arg_val[k]), chunk_size))]
        chunked_tensors.append(chunked_tensor)
    chunked_arg_val = [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]
    chunked_inputs = [{arg_key: c} for c in chunked_arg_val]

    return chunked_inputs

def get_dense_rep(x):
    """
    Get either qry_reps or tgt_reps.
    """
    if x["qry_reps"] is None and x["neg_reps"] is None:
        return x["tgt_reps"]
    elif x["tgt_reps"] is None and x["neg_reps"] is None:
        return x["qry_reps"]
    else:
        return x["neg_reps"]


@dataclass
class TrainTextImageDataCollator:
    data_args: DataArguments
    model_args: ModelArguments
    processor: ProcessorMixin

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        qry_inputs = self._get_batch_inputs(examples, "query_text", "query_image")
        pos_inputs = self._get_batch_inputs(examples, "pos_text", "pos_image")
        neg_inputs = self._get_batch_inputs(examples, "neg_text", "neg_image")
        return qry_inputs, pos_inputs

    def _get_batch_inputs(self, examples, text_keyname, image_keyname):
        texts, images = [], []
        for example in examples:
            # @ruimeng filter invalid data examples here will lead to fail to sync across devices (unequal batch size)
            # use dummy input for now
            if example is None or not example:
                text, image = '  ', None
            text, image = example[text_keyname], example[image_keyname]
            if type(text) == list:
                if len(text) == 0 or len(image) == 0:
                    text, image = '  ', None
                else:
                    text, image = text[0], image[0]
            texts.append(text)
            images.append(image)
        inputs = {'text': texts, 'image': images}
        return inputs

import os
@dataclass
class MultimodalDataCollator:
    processor: ProcessorMixin
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    batch_size: Optional[int] = None  # used to verify if a batch has invalid data
    ref_processor: Optional[ProcessorMixin] = None
    _is_skipping: Optional[bool] = False
    num_workers_per_node: Optional[int] = None

    def _get_example_inputs(self, text, raw_images):
        if not self._is_skipping and type(raw_images) == dict:
            visual_input = []
            assert 'resolutions' in raw_images, "we need len(raw_images['resolutions']) to determine the number of images, set it a list of None of for cases that no resizing is needed"
            num_images = len(raw_images['resolutions'])
            for image_idx in range(num_images):
                bytes = raw_images['bytes'][image_idx] if 'bytes' in raw_images else None
                path = raw_images['paths'][image_idx] if 'paths' in raw_images else None
                image_resolution = raw_images['resolutions'][image_idx] if 'resolutions' in raw_images else None
                if bytes is None and path is None:
                    image = None
                elif bytes is not None:
                    # vidore, image inputs are already bytes
                    image = Image.open(io.BytesIO(bytes))
                elif path is not None:
                    # mmeb/video datasets, lazy image loading and processing
                    with Image.open(path) as img:
                        image = img.convert("RGB")
                else:
                    print_rank(f"\n{'=' * 50}\nsomething went wrong with a data point from {example['global_dataset_name']}, neither bytes or path is given. \n\t\tquery_text: {example['query_text']}")
                if not self.data_args.resize_use_processor and image is not None and image_resolution:
                    image = image.resize(image_resolution)
                if image is not None and (self.data_args.image_decay_factor is not None and image_resolution is None):
                    assert image_resolution is None, "image_resolution is conflicting with image_decay_factor"
                    assert self.model_args.model_backbone in [QWEN2_5_VL, QWEN2_5_VL_COMPRESSION], "image_decay_factor is only supported for Qwen models"
                    # TODO: this is a hacky way to decay image resolution, need to be refactored
                    max_pixels = max(self.data_args.resize_min_pixels, self.data_args.resize_max_pixels * self.data_args.image_decay_factor ** (num_images - image_idx))
                    width, height = image.size
                    resized_height, resized_width = smart_resize(
                        height,
                        width,
                        min_pixels=self.data_args.resize_min_pixels,
                        max_pixels=max_pixels,
                    )
                    image = image.resize((resized_width, resized_height))  
                visual_input.append(image)
        else:
            visual_input = None
        return text, visual_input
        
    def _get_batch_inputs(self, batch, text_keyname, image_keyname):
        texts, visual_inputs = [], []
        for example in batch:
            # @ruimeng filter invalid data examples here may lead to fail to sync across devices (unequal batch size)
            # use dummy input for now
            if example is None or not example:
                texts.append('  ')
                visual_inputs.append(None)
            else:
                text, raw_images = example[text_keyname], example[image_keyname] if image_keyname in example else None

                text, visual_input = self._get_example_inputs(text, raw_images)
                texts.append(text)
                visual_inputs.append(visual_input)

        inputs = {'text': texts, 'images': visual_inputs}
        return inputs

    def _get_ret_batch_inputs(self, examples):
        """
        :param examples: 'query_text', 'query_image_path', 'pos_text', 'pos_image_path', 'neg_text', 'neg_image_path'
        """
        qry_inputs = self._get_batch_inputs(examples, "query_text", "query_image")
        pos_inputs = self._get_batch_inputs(examples, "pos_text", "pos_image")
        bs = len(qry_inputs['text'])
        assert bs > 0, 'An empty batch'
        # pad batch to batch_size to avoid hanging in distributed training
        if self.batch_size is not None and bs < self.batch_size:
            raise RuntimeError(f"Expect batch size {self.batch_size}, but got batch size of {bs}")
        process_fn = process_vlm_inputs_fns[self.training_args.model_backbone]
        if not callable(process_fn):
            process_fn = process_fn["retrieval"]
        processed_qry_inputs = process_fn(qry_inputs, processor=self.processor, max_length=self.data_args.max_len)
        processed_pos_inputs = process_fn(pos_inputs, processor=self.processor, max_length=self.data_args.max_len)

        # print_rank("inputs: " + str({ k: v.shape for k, v in processed_qry_inputs.items() if isinstance(v, torch.Tensor) }))
        # print_rank("inputs: " + str({ k: type(v) for k, v in processed_qry_inputs.items() if not isinstance(v, torch.Tensor) }))

        return { "qry": processed_qry_inputs, "tgt": processed_pos_inputs }
    
    def _constuct_conversations(self, enc_qry_inputs, enc_ans_inputs, dec_qry_inputs, dec_ans_inputs):
        conversations = []
        enc_texts = []
        for enc_qry_text, enc_ans_text, dec_qry_texts, dec_ans_texts in \
            zip(enc_qry_inputs['text'], enc_ans_inputs['text'], dec_qry_inputs['text'], dec_ans_inputs['text']):
            enc_texts.append(enc_qry_text + enc_ans_text if enc_qry_text is not None and enc_ans_text is not None else "")
            assert type(dec_qry_texts) == list and type(dec_ans_texts) == list
            assert len(dec_qry_texts) == len(dec_ans_texts)
            conversation = []
            for dec_qry_text, dec_ans_text in zip(dec_qry_texts, dec_ans_texts):
                conversation.extend([
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": dec_qry_text}],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": dec_ans_text}],
                    }
                ])
            conversations.append(conversation)
        texts = self.processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
        texts = [enc_t + t for enc_t, t in zip(enc_texts, texts)]  # filter out empty queries
        return {"text": texts, "images": enc_qry_inputs['images']}

    def _get_qa_batch_inputs(self, examples):
        """
        :param examples: 'encode_query', 'encode_answer', 'encode_image', 'decode_query', 'decode_answer'
        """
        enc_qry_inputs = self._get_batch_inputs(examples, "encode_query", "encode_image")
        enc_ans_inputs = self._get_batch_inputs(examples, "encode_answer", "")
        dec_qry_inputs = self._get_batch_inputs(examples, "decode_query", "")
        dec_ans_inputs = self._get_batch_inputs(examples, "decode_answer", "")
        inputs = self._constuct_conversations(enc_qry_inputs, enc_ans_inputs, dec_qry_inputs, dec_ans_inputs)
        bs = len(enc_qry_inputs['text'])
        assert bs > 0, 'An empty batch'
        # pad batch to batch_size to avoid hanging in distributed training
        if self.batch_size is not None and bs < self.batch_size:
            raise RuntimeError(f"Expect batch size {self.batch_size}, but got batch size of {bs}")
        process_fn = process_vlm_inputs_fns[self.training_args.model_backbone]
        if not callable(process_fn):
            process_fn = process_fn["openqa"]

        if self.ref_processor is not None:
            inputs['ref_texts'] = [self.ref_processor.apply_chat_template(eval(e['raw_conversations']), tokenize=False, add_generation_prompt=False) for e in examples]

        processed_inputs = process_fn(
            inputs, 
            processor=self.processor, 
            ref_processor=self.ref_processor, 
            max_length=self.data_args.max_len,
        )

        print_rank("inputs: " + str({ k: v.shape for k, v in processed_inputs.items() if isinstance(v, torch.Tensor) }))
        print_rank("inputs: " + str({ k: type(v) for k, v in processed_inputs.items() if not isinstance(v, torch.Tensor) }))

        return processed_inputs


    def __call__(self, examples):
        task_type = examples[0].get('task_type', 'retrieval')  # we assert task_type is the same for all examples in a batch
        assert all(x['task_type'] == task_type for x in examples), \
            f"Task type mismatch in the batch: {[e['task_type'] for e in examples]}"
        if task_type == 'retrieval':
            return self._get_ret_batch_inputs(examples)
        elif task_type == 'openqa':
            return self._get_qa_batch_inputs(examples)
        else:
            raise NotImplementedError(f"Unsupported task type [{task_type}] in the collator.")
