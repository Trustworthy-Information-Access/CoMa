import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s')
logger = logging.getLogger(__name__)
import torch
import os

def print_rank(message):
    """If distributed is initialized, print the rank."""
    if torch.distributed.is_initialized():
        logger.info(f'rank{torch.distributed.get_rank()}: ' + message)
    else:
        logger.info(message)


def print_master(message):
    """If distributed is initialized print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            logger.info(message)
    else:
        logger.info(message)


def find_latest_checkpoint(output_dir):
    """ Scan the output directory and return the latest checkpoint path """
    if not os.path.exists(output_dir):
        return None

    checkpoints = [
        os.path.join(output_dir, d) for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]

    if not checkpoints:
        return None

    # Sort by checkpoint number and return the latest one
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
    return latest_checkpoint


def batch_to_device(batch, device):
    _batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            _batch[key] = value.to(device)
        else:
            _batch[key] = value
    return _batch


def visual_attention_masks(mask_list, sep_indices, name):
    import matplotlib.pyplot as plt
    import json
    
    if not mask_list:
        return
    if sep_indices is not None:
        sep_indices = sep_indices[0].tolist()

    if any(mask is None for mask in mask_list):
        if torch.distributed.get_rank() == 0:
            breakpoint()
    torch.distributed.barrier()
    # Process masks: extract first batch element and convert to visibility (1=visible, 0=masked)
    masks = [(mask[0, 0] == 0).float().cpu().numpy() for mask in mask_list]

    # Setup grid
    cols = min(6, len(masks))
    rows = (len(masks) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
    axes = [axes] if rows == cols == 1 else axes.flatten() if rows > 1 or cols > 1 else [axes]
    
    # Plot masks
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, mask in enumerate(masks):
        axes[i].imshow(mask, cmap='viridis', aspect='auto')
        axes[i].set_title(f'Layer {i}')
        axes[i].axis('off')
        if sep_indices is not None:
            for j, idx in enumerate(sep_indices):
                if idx == 0: continue
                color = colors[j % len(colors)]
                axes[i].axvline(x=idx, color=color, linestyle='--', linewidth=1.5)
                axes[i].axhline(y=idx, color=color, linestyle='--', linewidth=1.5)

    # Hide unused subplots
    for i in range(len(masks), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"attnmask_layers_{name}.png", dpi=150, bbox_inches='tight')
    plt.close()

    with open(f"attnmask_sep_{name}.json", 'w') as f:
        json.dump(sep_indices, f, indent=4)