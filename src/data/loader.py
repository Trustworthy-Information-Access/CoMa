import torch
from typing import Dict, List
from datasets.distributed import split_dataset_by_node
from torch.utils.data import IterableDataset

from src.data.dataset.base_pair_dataset import AutoPairDataset
from src.data.dataset.base_qa_dataset import AutoQADataset
from src.data.dataset.hf_datasets import interleave_datasets
from src.utils import print_master

class CustomDataLoader:
    def __init__(self, dataset: IterableDataset, batch_size, collate_fn=None):
        self.batch_size = batch_size
        self.dataset = dataset
        self.collate_fn = collate_fn
    
    def __iter__(self):
        batch = []
        # 从数据集中逐个获取样本
        for sample in self.dataset:
            batch.append(sample)
            # 当批次大小达到预设值时，进行 collate 并 yield
            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                # 清空 batch 列表以供下一批次使用
                batch = []

        if batch:
            yield self.collate_fn(batch)

    def __len__(self):
        if not hasattr(self.dataset, '__len__'):
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
        if hasattr(self.dataset, 'num_rows'):
            return (self.dataset.num_rows + self.batch_size - 1) // self.batch_size
        raise TypeError("Dataset does not support __len__ or num_rows")

def init_retrieval_dataloader(dataset_config, model_args, data_args, training_args, data_collator=None):

    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    probs = _calculate_probabilities(dataset_config, "retrieval")
    train_datasets = _create_datasets(dataset_config, 'retrieval', model_args, data_args, training_args, probs)

    interleave_batch_size = _get_interleave_batch_size(training_args, world_size)

    total_num_rows = _validate_datasets(train_datasets, training_args.per_device_train_batch_size * world_size, 'retrieval')

    print_master(
        f"\nInitializing interleaved datasets:"
        f"\n\t\tworld_size={world_size}"
        f"\n\t\ttotal_num_rows={total_num_rows}"
        f"\n\t\tglobal_batch_size={training_args.per_device_train_batch_size * world_size}"
        f"\n\t\tinterleave_batch_size={interleave_batch_size}"
    )

    train_dataset = _create_interleaved_dataset(train_datasets, probs, interleave_batch_size, training_args)
    
    if torch.distributed.is_initialized():
        train_dataset = split_dataset_by_node(train_dataset, rank=torch.distributed.get_rank(), world_size=world_size) if train_datasets else None
        if train_dataset:
            setattr(train_dataset, 'num_rows', total_num_rows // world_size)
    return CustomDataLoader(
        dataset=train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=data_collator
    )

def init_openqa_dataloader(dataset_config, model_args, data_args, training_args, data_collator=None):
    
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    
    probs = _calculate_probabilities(dataset_config, "openqa")
    train_datasets = _create_datasets(dataset_config, 'openqa', model_args, data_args, training_args)

    total_num_rows = _validate_datasets(train_datasets, training_args.per_device_train_batch_size * world_size, 'openqa')

    print_master(
        f"\nInitializing interleaved datasets:"
        f"\n\t\tworld_size={world_size}"
        f"\n\t\ttotal_num_rows={total_num_rows}"
        f"\n\t\tglobal_batch_size={training_args.per_device_train_batch_size * world_size}"
    )

    # No interleaving for openqa
    assert len(train_datasets) == 1, "OpenQA dataset should not be interleaved"
    train_dataset = _create_interleaved_dataset(train_datasets, probs, training_args.per_device_train_batch_size * world_size, training_args)

    if torch.distributed.is_initialized():
        train_dataset = split_dataset_by_node(train_dataset, rank=torch.distributed.get_rank(), world_size=world_size) if train_dataset else None
    return CustomDataLoader(
        dataset=train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=data_collator
    )

def _validate_datasets(datasets, global_batch_size, task_type):
    """Validate that datasets have sufficient rows."""
    total_rows = sum(d.num_rows for d in datasets)
    if total_rows == 0:
        return total_rows

    assert total_rows >= global_batch_size, (
        f"total_num_rows(={total_rows}) for {task_type} must be >= global batch size "
        f"(={global_batch_size}), since the last batch will be dropped."
    )
    return total_rows

def _calculate_probabilities(dataset_config, task_type="retrieval"):
    weights = [d['weight'] for d in dataset_config.values() if d.get('task_type', 'retrieval') == task_type]
    weight_sum = sum(weights)
    return [w / weight_sum for w in weights] if weight_sum > 0 else []

def _create_datasets(dataset_config, task_type, model_args, data_args, training_args, probs=None):
    """Create datasets for a specific task type."""
    datasets = []
    dataset_idx = 0
    print_master(f"=========== Create {task_type.upper()} datasets ===========")
    for data_idx, (global_dataset_name, config) in enumerate(dataset_config.items()):
        if config.get('task_type', 'retrieval') != task_type:
            continue
            
        if task_type == 'retrieval':
            dataset = AutoPairDataset.instantiate(model_args=model_args, data_args=data_args, training_args=training_args, **config)
        elif task_type == 'openqa':
            dataset = AutoQADataset.instantiate(model_args=model_args, data_args=data_args, training_args=training_args, **config)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
            
        datasets.append(dataset)
        print_master(
            f"\t\tDataset#{data_idx} (dataset_parser={config.get('dataset_parser', 'n/a')}): "
            f"{global_dataset_name}, task_type={task_type}, num_rows={dataset.num_rows}, "
            f"prob={probs[dataset_idx] * 100.0:.2f}%" if probs is not None else ""
        )
        dataset_idx += 1
    print_master(f"Total {len(datasets)} {task_type.upper()} datasets created.")
    
    return datasets

def _get_interleave_batch_size(training_args, world_size):
    """Calculate interleave batch size."""
    
    if training_args.interleave_batch_size and training_args.interleave_batch_size <= 1.0:
        return int(training_args.per_device_retrv_train_batch_size * world_size * training_args.interleave_batch_size)
    return training_args.interleave_batch_size

def _create_interleaved_dataset(datasets, probs, batch_size, training_args):
    """Create interleaved dataset if multiple datasets exist."""
    if len(datasets) > 1:
        return interleave_datasets(
            datasets, probabilities=probs, batch_size=batch_size,
            seed=training_args.seed, stopping_strategy=training_args.interleave_stopping_strategy
        )
    return datasets[0] if datasets else None