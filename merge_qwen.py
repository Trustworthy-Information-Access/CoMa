from transformers import AutoConfig, AutoTokenizer
import torch
import os
from peft import PeftModel, LoraConfig

from src.model.vlm_backbone.qwen2_5_vl_compression import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

import argparse

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('model_name_or_path', type=str)
    parser.add_argument('--base_model_name_or_path', type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")

    args = parser.parse_args()

    base_model_path = args.base_model_name_or_path
    model_name_or_path = args.model_name_or_path
    save_path = os.path.join(model_name_or_path, 'merged')
    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    # config._attn_implementation = "sdpa"
    config.padding_side = "left"
    config.use_cache = False
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    lora_config = LoraConfig.from_pretrained(model_name_or_path)
    lora_model = PeftModel.from_pretrained(base_model, model_name_or_path, config=lora_config, is_trainable=False)
    lora_model.load_adapter(model_name_or_path, lora_model.active_adapter, is_trainable=False)
    if config.tie_word_embeddings:
        print("Tieing weights...")
        lora_model.model.lm_head._parameters['weight'] = lora_model.model.model.embed_tokens.weight
        assert lora_model.model.lm_head.weight.data_ptr() == lora_model.model.model.embed_tokens.weight.data_ptr(), "Failed to tie weights"

    model = lora_model.merge_and_unload()

    model.save_pretrained(save_path)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        processor = Qwen2_5_VLProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        processor = Qwen2_5_VLProcessor.from_pretrained(base_model_path, trust_remote_code=True)

    tokenizer.save_pretrained(save_path)
    processor.save_pretrained(save_path)

if __name__ == '__main__':
    main()