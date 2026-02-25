import os
from typing import Dict
import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel

from src.arguments import ModelArguments
from src.model.processor import get_backbone_name, backbone2model, print_master, VLM_IMAGE_TOKENS, QWEN2_5_VL, QWEN2_5_VL_COMPRESSION, TOTAL_COMPRESS_TOKENS
from src.model.modules import GatedPoolingSwiGLU, LatentAttentionLayer


class MMEBModel(nn.Module):
    TRANSFORMER_CLS = AutoModelForCausalLM

    def __init__(self,
        encoder: PreTrainedModel,
        ref_model: PreTrainedModel = None,
        ref_processor = None,
        projector: nn.Module = None,
        pooling: str = 'last',
        normalize: bool = False,
        temperature: float = 0.02,
        training_args=None,
        model_args=None,
    ):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.config = encoder.config
        self.encoder = encoder
        self.ref_model = ref_model
        self.ref_processor = ref_processor
        self.projector = projector
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def encode_input(self, input, **kwargs):
        hidden_states = self.encoder(**input, return_dict=True, output_hidden_states=True, output_attentions=kwargs.get('output_attentions', False))
        hidden_states = hidden_states.hidden_states[-1]
        pooled_output = self._pooling(hidden_states, input['attention_mask'])
        return pooled_output

    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling == 'last' or self.pooling == 'eos':
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            batch_size = last_hidden_state.shape[0]
            if left_padding:
                # Get the vectors at the last position
                reps = last_hidden_state[torch.arange(batch_size), -1, :]
            else:
                # Calculate last 1 position in the original tensor
                eos_indices = attention_mask.sum(dim=1) - 1
                # Get the vectors at the last 1 position of each attention mask
                reps = last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device), eos_indices]
        elif self.pooling == 'avg':
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            batch_size = last_hidden_state.shape[0]
            if left_padding:
                # Get the vectors at the average position
                reps = last_hidden_state[torch.arange(batch_size), -TOTAL_COMPRESS_TOKENS:, :].mean(dim=1)
            else:
                # Calculate average TOTAL_COMPRESS_TOKENS position in the original tensor
                eos_indices = attention_mask.sum(dim=1)
                token_indices = torch.arange(TOTAL_COMPRESS_TOKENS, device=last_hidden_state.device).unsqueeze(0)  # [1, T]
                token_indices = token_indices + eos_indices.unsqueeze(1) - TOTAL_COMPRESS_TOKENS  # [B, T]
                batch_indices = torch.arange(batch_size, device=last_hidden_state.device).unsqueeze(1).expand(-1, TOTAL_COMPRESS_TOKENS)  # [B, T]
                # Get the vectors at the average TOTAL_COMPRESS_TOKENS position of each attention mask
                reps = last_hidden_state[batch_indices, token_indices].mean(dim=1)
        elif self.pooling == 'concat':
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            batch_size = last_hidden_state.shape[0]
            if left_padding:
                # Get the vectors at the concat position
                reps = last_hidden_state[torch.arange(batch_size), -TOTAL_COMPRESS_TOKENS:, :].contiguous().view(batch_size, -1)  # [B, T * D]
            else:
                # Calculate concat TOTAL_COMPRESS_TOKENS position in the original tensor
                eos_indices = attention_mask.sum(dim=1)
                start_indices = eos_indices - TOTAL_COMPRESS_TOKENS
                token_indices = torch.arange(TOTAL_COMPRESS_TOKENS, device=last_hidden_state.device).unsqueeze(0)  # [1, T]
                token_indices = token_indices + start_indices.unsqueeze(1)  # [B, T]
                batch_indices = torch.arange(batch_size, device=last_hidden_state.device).unsqueeze(1).expand(-1, TOTAL_COMPRESS_TOKENS)  # [B, T]
                # Get the vectors at the concat TOTAL_COMPRESS_TOKENS position of each attention mask
                reps = last_hidden_state[batch_indices, token_indices].contiguous().view(batch_size, -1)  # [B, T * D]
        elif self.pooling in {'proj', 'attn'}:
            assert self.projector is not None, "Projector must be provided for projector fusion"
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            batch_size = last_hidden_state.shape[0]
            if left_padding:
                # Get the vectors at the compressed position
                reps = last_hidden_state[torch.arange(batch_size), -TOTAL_COMPRESS_TOKENS:, :]  # [B, T, D]
            else:
                # Calculate TOTAL_COMPRESS_TOKENS position in the original tensor
                eos_indices = attention_mask.sum(dim=1)
                start_indices = eos_indices - TOTAL_COMPRESS_TOKENS
                token_indices = torch.arange(TOTAL_COMPRESS_TOKENS, device=last_hidden_state.device).unsqueeze(0)  # [1, T]
                token_indices = token_indices + start_indices.unsqueeze(1)  # [B, T]
                batch_indices = torch.arange(batch_size, device=last_hidden_state.device).unsqueeze(1).expand(-1, TOTAL_COMPRESS_TOKENS)  # [B, T]
                # Get the vectors at the concat TOTAL_COMPRESS_TOKENS position of each attention mask
                reps = last_hidden_state[batch_indices, token_indices]  # [B, T, D]
            reps = self.projector(reps)  # [B, D]
        else:
            raise NotImplementedError
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    @classmethod
    def build(cls, model_args: ModelArguments, training_args, **kwargs):
        config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
        model_backbone = get_backbone_name(hf_config=config, model_type=model_args.model_type)
        projector=None
        print_master(f'Loading backbone [{model_backbone}] from {model_args.model_name}')
            
        # Loading the base model
        if model_backbone == QWEN2_5_VL_COMPRESSION:
            print_master(f"** Total Compress Tokens: {TOTAL_COMPRESS_TOKENS} **")

            config._attn_implementation = "sdpa"
            config.padding_side = "left"
            config.use_cache = False
            base_model = backbone2model[model_backbone].from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            if not model_args.lora:
                print_master(f'** Training MLP and LLM for {model_backbone} **')
                for p in base_model.visual.parameters():
                    p.requires_grad = False  # freeze vision encoder
                for p in base_model.visual.merger.parameters():
                    p.requires_grad = True  # mlp
                for p in base_model.model.parameters():
                    p.requires_grad = True  # llm
                base_model.lm_head.requires_grad = True  # llm head
            if model_args.pooling in {'proj', 'attn'}:
                if model_args.pooling == 'proj':
                    projector = GatedPoolingSwiGLU(config.hidden_size, config.hidden_size)
                elif model_args.pooling == 'attn':
                    projector = LatentAttentionLayer(config.hidden_size, config.hidden_size, num_latents=max(1, TOTAL_COMPRESS_TOKENS // 4))
                print_master(f'Loading projector for pooling: {projector}')
        else:
            config.use_cache = False
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                model_args.model_name, **kwargs, config=config,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True)

        if model_args.ref_model_name:
            # TODO: hard coded
            from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
            ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_args.ref_model_name, **kwargs, config=config,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True)
            ref_processor = Qwen2_5_VLProcessor.from_pretrained(model_args.ref_model_name)
        else:
            ref_model, ref_processor = None, None

        if model_args.lora:
            print_master(f'Loading lora adapter from {base_model}')
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=model_args.lora_target_modules.split(',') if "," in model_args.lora_target_modules else model_args.lora_target_modules,
                lora_dropout=model_args.lora_dropout,
                modules_to_save=["embed_tokens"] if model_backbone == QWEN2_5_VL_COMPRESSION and TOTAL_COMPRESS_TOKENS > 0 else None,  # finetune embeddings
                init_lora_weights="gaussian",
                use_dora=True,
                inference_mode=False
            )
            lora_model = get_peft_model(base_model, lora_config)
            if config.tie_word_embeddings:
                print(f"Tying weights for {model_backbone}")
                lora_model.model.lm_head._parameters['weight'] = lora_model.model.model.embed_tokens.weight  # HACK: tie weights manually
                assert lora_model.model.lm_head.weight.data_ptr() == lora_model.model.model.embed_tokens.weight.data_ptr(), "Failed to tie weights"

            lora_model.print_trainable_parameters()
            model = cls(
                encoder=lora_model,
                projector=projector,
                ref_model=ref_model,
                ref_processor=ref_processor,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                model_args=model_args,
                training_args=training_args,
            )
        else:
            model = cls(
                encoder=base_model,
                projector=projector,
                ref_model=ref_model,
                ref_processor=ref_processor,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                model_args=model_args,
                training_args=training_args,
            )
        return model


    @classmethod
    def load(cls, model_args: ModelArguments, is_trainable=True, **kwargs):
        # Loading the base model
        model_name_or_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        projector = None
        if not hasattr(model_args, "model_backbone") or not model_args.model_backbone:
            model_backbone = get_backbone_name(hf_config=config, model_type=model_args.model_type)
            setattr(model_args, 'model_backbone', model_backbone)
        print_master(f'Loading backbone [{model_args.model_backbone}] from {model_name_or_path}')
            
        if model_args.model_backbone == QWEN2_5_VL:
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config._attn_implementation = "flash_attention_2"
            config.vision_config._attn_implementation = "flash_attention_2"
            config.padding_side = "left"
            base_model = backbone2model[model_args.model_backbone].from_pretrained(
                model_args.model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                config=config
            )
        elif model_args.model_backbone == QWEN2_5_VL_COMPRESSION:
            print_master(f"** Total Compress Tokens: {TOTAL_COMPRESS_TOKENS} **")

            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config._attn_implementation = "sdpa"
            base_model = backbone2model[model_args.model_backbone].from_pretrained(
                model_args.model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                config=config
            )
            if not model_args.lora and is_trainable:
                print_master(f'** Training MLP and LLM for {model_args.model_backbone} **')
                for p in base_model.visual.parameters():
                    p.requires_grad = False  # freeze vision encoder
                for p in base_model.visual.merger.parameters():
                    p.requires_grad = True  # mlp
                for p in base_model.model.parameters():
                    p.requires_grad = True  # llm
                base_model.lm_head.requires_grad = True  # llm head
            if model_args.pooling in {'proj', 'attn'}:
                if model_args.pooling == 'proj':
                    projector = GatedPoolingSwiGLU(config.hidden_size, config.hidden_size)
                elif model_args.pooling == 'attn':
                    projector = LatentAttentionLayer(config.hidden_size, config.hidden_size, num_latents=max(1, TOTAL_COMPRESS_TOKENS // 4))
                projector_path = model_args.projector_path if model_args.projector_path else os.path.join(model_args.model_name, 'projector.pth')
                projector.load_state_dict(torch.load(projector_path, map_location='cpu'))
                print_master(f'Loading projector for pooling: {projector}')
        else:
            # Loading external base model from HF
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config.use_cache = False
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                model_name_or_path, **kwargs, config=config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True)

        # Building the model on top of the base
        if model_args.lora:
            print_master(f'Loading LoRA from {model_name_or_path}')
            lora_config = LoraConfig.from_pretrained(model_name_or_path)
            lora_model = PeftModel.from_pretrained(base_model, model_name_or_path, config=lora_config, is_trainable=is_trainable)
            lora_model.load_adapter(model_name_or_path, lora_model.active_adapter, is_trainable=is_trainable)
            if config.tie_word_embeddings:
                print_master("Tieing weights...")
                lora_model.model.lm_head._parameters['weight'] = lora_model.model.model.embed_tokens.weight  # HACK: tie weights manually
                assert lora_model.model.lm_head.weight.data_ptr() == lora_model.model.model.embed_tokens.weight.data_ptr(), "Failed to tie weights"
            if not is_trainable:
                lora_model = lora_model.merge_and_unload()
            model = cls(
                encoder=lora_model,
                projector=projector,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        else:
            model = cls(
                encoder=base_model,
                projector=projector,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )

        model.model_backbone = model_args.model_backbone
        return model

    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)

    def forward(self, **kwargs):
        qry = kwargs.pop('qry', None)
        tgt = kwargs.pop('tgt', None)
        neg = kwargs.pop('neg', None)

        if qry is None and tgt is None and neg is None:
            ref_kwargs = {
                k[len('ref_'):]: v
                for k, v in kwargs.items()
                if k.startswith('ref_')
            }
            for k in ref_kwargs:
                kwargs.pop(f'ref_{k}')
            if self.ref_model is not None:
                with torch.no_grad():
                    ref_logits = self.ref_model(**ref_kwargs).logits
            else:
                ref_logits = None
            output = self.encoder(**kwargs, ref_logits=ref_logits)  # openqa
            return output

        qry_reps = self.encode_input(qry, **kwargs) if qry else None  # (bsz_per_device, dim)
        tgt_reps = self.encode_input(tgt, **kwargs) if tgt else None # (bsz_per_device, dim)
        neg_reps = self.encode_input(neg, **kwargs) if neg else None # (num_neg * bsz_per_device, dim)

        if qry_reps is None or tgt_reps is None:
            return {"qry_reps": qry_reps, "tgt_reps": tgt_reps, "neg_reps": neg_reps}

        if self.is_ddp:
            all_qry_reps = self._dist_gather_tensor(qry_reps)
            all_tgt_reps = self._dist_gather_tensor(tgt_reps)
            all_neg_reps = self._dist_gather_tensor(neg_reps) if neg_reps is not None else None
        else:
            all_qry_reps = qry_reps
            all_tgt_reps = tgt_reps
            all_neg_reps = neg_reps

        batch_size = all_qry_reps.shape[0]
        # (bsz, bsz )
        scores = self.compute_similarity(all_qry_reps, all_tgt_reps)

        if neg_reps is not None:
            neg_ratio = int(all_neg_reps.shape[0] / all_qry_reps.shape[0])
            neg_scores = torch.sum(all_qry_reps.unsqueeze(1) * all_neg_reps.view(batch_size, neg_ratio, -1), dim = -1) # B * neg_ratio
            # (bsz, bsz + neg_ratio )
            scores = torch.cat([scores, neg_scores], dim = 1)

        if torch.distributed.get_rank() == 0:
            print("Scores", scores.shape)

        scores = scores.view(all_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_qry_reps.size(0) // all_tgt_reps.size(0))
        loss = self.cross_entropy(scores / self.temperature, target)
        if self.is_ddp:
            loss = loss * self.world_size

        return loss

    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs = None):
        gradient_checkpointing_kwargs={'use_reentrant': False}
        if self.model_args.lora:
            model = self.encoder.base_model.model
        else:
            model = self.encoder
        if self.training_args.bf16:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        else:
            model.gradient_checkpointing_enable()
    
