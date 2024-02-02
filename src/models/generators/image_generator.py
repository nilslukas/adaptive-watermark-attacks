from typing import Callable, Tuple

import torch
from torch import nn

from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs


class ImageGenerator(nn.Module):
    """ A vanilla image generator pipeline with the capacity to be watermarked. """

    def __init__(self, model_args: ModelArgs, env_args: EnvArgs = None):
        super().__init__()
        self.model_args = model_args
        self.env_args = env_args if env_args is not None else EnvArgs()
        self.wm_key = None

    def generate(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Generate an image given some input. Unconditional generation.
        @return images [n, c, h, w] in [0,1] and the latent vector [n, d]
        """
        raise NotImplementedError

    def set_watermarking_key(self, wm_key):
        """ Optional: Set a watermarking key during generation to apply watermarking. """
        self.wm_key = wm_key

    def save(self, **kwargs) -> dict:
        """
        Saves the generator to disk and returns its state dict
        """
        raise NotImplementedError

    def load(self, **kwargs) -> 'ImageGenerator':
        """
        Load the generator from disk and return the loaded generator
        """
        raise NotImplementedError

    def add_post_processing(self, post_processing_fn: Callable):
        """
         Adds a post-processing hook to each generated image
        """
        raise NotImplementedError

    def clear_post_processing(self):
        """
        Clears all post-processing hooks to each generated image
        """
        raise NotImplementedError
