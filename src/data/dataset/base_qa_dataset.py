from abc import ABCMeta, abstractmethod
from functools import wraps
from datasets import Features, Value, Sequence


MULTIMODAL_FEATURES = Features(**{
    "encode_query": Value(dtype='string'),
    "encode_answer": Value(dtype='string'),
    "encode_image": {
        "paths": Sequence(Value(dtype='string')),  # List of image paths (frames)
        "bytes": Sequence(Value(dtype='binary')),  # List of pre-saved image bytes
        "resolutions": Sequence(Sequence(Value(dtype='int32'), length=2))  # List of [width, height] pairs
    },
    "decode_query": Sequence(Value(dtype='string')),
    "decode_answer": Sequence(Value(dtype='string')),
    "global_dataset_name": Value(dtype='string'),
    "task_type": Value(dtype='string'),
    "raw_conversations": Value(dtype='string'),
    "compression": {
        "image": Value(dtype='string'),
        "query": Value(dtype='string'),
    },
})

RESOLUTION_MAPPING = {
    "high": (1344, 1344),
    "mid": (672, 672),
    "low": (128, 128),
}


class AutoQADataset(metaclass=ABCMeta):
    # Base class for auto datasets.
    registry = {}

    def __init_subclass__(cls):
        if cls.__name__ not in AutoQADataset.registry:
            AutoQADataset.registry[cls.__name__] = cls
        else:
            raise RuntimeError('Subclass "{cls.__name__}" has already defined.')

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_config(config)` methods."
        )

    @classmethod
    def instantiate(cls, dataset_parser, *args, **kwargs):
        try:
            return cls.registry[dataset_parser](*args, **kwargs)
        except Exception as e:
            raise e

    @classmethod
    def register(cls, dataset_name):
        def inner_wrapper(wrapped_class):
            if dataset_name in cls.registry:
                print(f"[Alert] AutoQADataset: a class in the same name ({dataset_name}) has been registered")
            else:
                cls.registry[dataset_name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    @abstractmethod
    def main(self):
        pass

def add_metainfo_hook(f):
    """
    A post-processing wrapper function that add meta information (e.g. data_type, dataset_name, loss_type) into batches
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        # go through data pipeline customized to each dataset
        batch_data = f(*args, **kwargs)
        # append common metadata
        batch_size = len(batch_data['encode_query'])
        global_dataset_name = kwargs.get("global_dataset_name", "None")
        task_type = kwargs.get("task_type", "openqa")
        batch_data['global_dataset_name'] = [global_dataset_name] * batch_size
        batch_data['task_type'] = [task_type] * batch_size
        return batch_data

    return wrapper
