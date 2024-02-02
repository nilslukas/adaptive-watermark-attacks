from dataclasses import dataclass, field

from accelerate import Accelerator
from torch.utils.data import DataLoader

CACHE_DIR = "../.cache"


@dataclass
class EnvArgs:
    CONFIG_KEY = "env_args"
    """ This class contains all arguments for the environment where to load samples. """

    logging_tool: str = field(default=None, metadata={
        "help": "tool to log experimental data with. Currently, only wandb is supported. ",
        "choices": [None, "wandb"]
    })

    seed: int = field(default=1337, metadata={
        "help": "seed for randomness"
    })

    log_every: int = field(default=100, metadata={
        "help": "log interval for training"
    })

    save_every: int = field(default=249, metadata={
        "help": "save interval for training"
    })

    num_workers: int = field(default=16, metadata={
        "help": "number of workers for dataloading"
    })

    batch_size: int = field(default=16, metadata={
        "help": "default batch size for training"
    })

    eval_batch_size: int = field(default=128, metadata={
        "help": "default batch size for inference"
    })

    verbose: bool = field(default=False, metadata={
        "help": "whether to print out to the cmd line"
    })

    gradient_accumulation_steps: int = field(default=1, metadata={
        "help": "number of steps to accumulate gradients"
    })

    cache_modulation: bool = field(default=False, metadata={
        "help": "whether to use caching when computing weight modulation. Currently breaks gradients."
    })

    @property
    def device(self):
        return self.get_accelerator().device

    def get_accelerator(self):
        if not hasattr(self, "accelerator"):
            self.accelerator = Accelerator(gradient_accumulation_steps=1)
        return self.accelerator

    def make_data_loader(self, dataset, shuffle=True, batch_size=None) -> DataLoader:
        batch_size = self.batch_size if batch_size is None else batch_size
        dl = DataLoader(dataset, shuffle=shuffle, num_workers=self.num_workers, batch_size=batch_size)
        return self.get_accelerator().prepare(dl)
