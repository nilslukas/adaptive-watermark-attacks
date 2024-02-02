import os.path
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch

from src.utils.highlited_print import bcolors
from src.utils.web import download_and_unzip


@dataclass
class WatermarkingKeyArgs:
    CONFIG_KEY = "watermarking_key_args"
    """ 
    This class contains all arguments for the keygen procedure to generate watermarking keys. 
    """

    WM_KEY_ARGS_KEY = "watermarking_args_key"  # field to save in the dictionary.

    name: str = field(default=None, metadata={
        "help": "the name of the watermarking method."
                "PTW: Pivotal Tuning Watermarking (Ours)"
                "TRW: Tree Ring Watermark (Wen et al., https://arxiv.org/abs/2305.20030)"
                "WDM: Watermark DM (Zhao et al., https://arxiv.org/pdf/2303.10137.pdf)",
        "choices": ["ptw", "trw", "wdm", "sdw"]
    })

    double_tail: bool = field(default=False, metadata={
        "help": "whether to use double tail detection (only applicable to multi-bit watermark)"
    })

    @staticmethod
    def from_checkpoint(**kwargs) -> dict:
        """
        Override given values with a value from the checkpoint
        """
        key_ckpt = kwargs.setdefault('key_ckpt', None)
        key_ckpt = download_and_unzip(key_ckpt)

        if key_ckpt is not None and os.path.exists(key_ckpt):  # attempt to load your own watermarking params
            data = torch.load(key_ckpt)
            print(f"> Restoring watermark arguments from '{bcolors.OKGREEN}{os.path.abspath(key_ckpt)}{bcolors.ENDC}'")
            kwargs.update(vars(data[WatermarkingKeyArgs.WM_KEY_ARGS_KEY]))
            kwargs['key_ckpt'] = key_ckpt
        return kwargs

    @staticmethod
    def load_specific_class(**kwargs):
        return {
            "ptw": PTWWatermarkingKeyArgs,
            "trw": TRWWatermarkingKeyArgs,
            "wdm": WDMWatermarkingKeyArgs,
            "sdw": SDWWatermarkingKeyArgs,
            "plain": WatermarkingKeyArgs
        }[kwargs.setdefault("name", "plain")]

    optimizer: str = field(default="adam", metadata={
        "help": "the name of the optimizer",
        "choices": ["sgd", "adam"]
    })

    key_ckpt: str = field(default=None, metadata={
        "help": "path to a pre-trained key."
    })

    add_preprocessing: bool = field(default=False, metadata={
        "help": "add pre-processing to the key generation"
    })

    message: torch.Tensor = field(default=None, metadata={
        "help": "the message to embed"
    })

    pre_generation_folder: str = field(default=None, metadata={
        "help": "folder to store pre_generated images with their respective input noise and seed used to generate them."
                "If None, the images will be generated on the fly."
    })

    num_pre_generate: int = field(default=int(1e2), metadata={
        "help": "how many samples to pre_generate if pre_generation_folder is not None."
    })

    re_use_pre_generated: bool = field(default=False, metadata={
        "help": "whether to re-iterate through pre generated images when we run out of them or to start generating on"
                "the fly when we run out. Generate on the fly by default."
    })


@dataclass
class TRWWatermarkingKeyArgs(WatermarkingKeyArgs):
    # we will infuse these args into the watermarking_key_args class

    reversal_inference_steps: int = field(default=20, metadata={
        "help": "number of steps for the reversal inference"
    })

    w_channel: str = field(default=3, metadata={
        "help": "number of channels to use for the watermark. "
    })

    w_pattern: str = field(default='ring', metadata={
        "help": "pattern to use for the watermark. "
    })

    w_mask_shape: str = field(default='circle', metadata={
        "help": "mask shape to use for the watermark. "
    })

    w_radius: str = field(default=10, metadata={
        "help": "radius to use for the watermark. "
    })

    w_measurement: str = field(default='pval_complex', metadata={
        "help": "measurement to use for the watermark. "
    })

    w_injection: str = field(default='complex', metadata={
        "help": "injection to use for the watermark. "
    })

    w_pattern_const: str = field(default=0.0, metadata={
        "help": "pattern const to use for the watermark. "
    })

    w_seed: int = field(default=0, metadata={
        "help": "seed to use for the watermark. "
    })

    inversal_model: str = field(default=None, metadata={
        "help": "path to the inversal model. "
    })

    disable: bool = field(default=False, metadata={
        "help": "Load TRW model but disable watermarking"
    })


@dataclass
class PTWWatermarkingKeyArgs(WatermarkingKeyArgs):
    PREFIX = "ptw"  # every parameter name will have this prefix. (e.g., ptw_w_channel)

    # The PTW watermark predicts binary messages and thus needs an alphabet.
    ALPHABET = " ABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%"  # alphabet to convert messages from bits to string

    def __post_init__(self):
        if self.message is None:  # initialize a random message.
            self.message = ''.join(np.random.choice(list(PTWWatermarkingKeyArgs.ALPHABET), size=100))

    message: str = field(default=None, metadata={
        "help": "the embedded_message to embed. Will be converted into a (truncated) "
                "binary sequence. "
    })

    first_wm_layer: int = field(default=0, metadata={
        "help": "skip this many layers"
    })

    bitlen: int = field(default=50, metadata={
        "help": "bit length of the watermark."
    })

    weight_mapper: bool = field(default=False, metadata={
        "help": "whether to add a style mapper to the input. "
    })

    bias_mapper: bool = field(default=False, metadata={
        "help": "whether to add a bias mapper to the deconvolution layers"
    })

    style_mapper: bool = field(default=False, metadata={
        "help": "whether to add a style mapper to the input. "
    })

    lr_mapper: float = field(default=0.001, metadata={
        "help": "learning rate for the mapper"
    })

    lr_decoder: float = field(default=0.0001, metadata={
        "help": "learning rate for the detector"
    })

    lr_encoder: float = field(default=0.0001, metadata={
        "help": "learning rate for the encoder in the autoencoder"
    })

    keygen_steps: List[int] = field(default_factory=lambda: [1], metadata={
        "help": "activate gradients at these steps of the diffusion process."
    })

    keygen_lambda_lpips: float = field(default=0.01, metadata={
        "help": "lambda for the lpips loss"
    })

    pre_generation_folder: str = field(default=None, metadata={
        "help": "folder to store pre_generated images with their respective input noise and seed used to generate them."
                "If None, the images will be generated on the fly."
    })

    num_pre_generate: int = field(default=int(1e2), metadata={
        "help": "how many samples to pre_generate if pre_generation_folder is not None."
    })

    re_use_pre_generated: bool = field(default=False, metadata={
        "help": "whether to re-iterate through pre generated images when we run out of them or to start generating on"
                "the fly when we run out. Generate on the fly by default."
    })

    first_marked_layer: int = field(default=0, metadata={
        "help": "the first watermarked layer (0=all layers will be watermarked)"
    })

    decoder_arch: str = field(default="resnet18", metadata={
        "help": "model architecture of the decoder",
        "choices": ["resnet18", "resnet50", "resnet101"]
    })


@dataclass
class WDMWatermarkingKeyArgs(WatermarkingKeyArgs):
    bitlen: int = field(default=40, metadata={
        "help": "number of bits to embed"
    })

    decoder_arch: str = field(default="resnet18", metadata={
        "help": "model architecture of the decoder",
        "choices": ["resnet18", "resnet50", "resnet101"]
    })

    starter_keygen_lambda_lpips: float = field(default=0.01, metadata={
        "help": "if other than keygen_lambda_lpips, it will be the coef used before we reach the trigger capacity"
    })

    keygen_lambda_lpips: float = field(default=0.0, metadata={
        "help": "lambda for the lpips loss"
    })

    l1_loss_coef: float = field(default=0.1, metadata={
        "help": "lambda for the l1 loss"
    })

    heatup_period: int = field(default=10, metadata={
        "help": ""
    })

    l_inf_loss_coef: float = field(default=0.1, metadata={
        "help": ""
    })

    l_inf_epsilon: float = field(default=0.03125, metadata={
        "help": ""
    })

    capacity_trigger: int = field(default=10, metadata={
        "help": ""
    })

    lr: float = field(default=0.001, metadata={
        "help": "learning rate for the surrogate decoder"
    })



@dataclass
class SDWWatermarkingKeyArgs(WatermarkingKeyArgs):
    DECODER_STATE_DICT_KEY = "decoder_state_dict"

    bitlen: int = field(default=40, metadata={
        "help": "number of different classes (as a power of 2)"
    })

    lr: float = field(default=0.001, metadata={
        "help": "learning rate for the surrogate decoder"
    })

    dataset_root: str = field(default=None, metadata={
        "help": "path to a flat dataset with unwatermarked images. Optional, just to speed up keygen "
    })

    resume_from: str = field(default=None, metadata={
        "help": "path to a pre-trained key."
    })

    wm_method: str = field(default="dwtdct", metadata={
        "help": "method to use for watermarking",
        "choices": ["dwtdct", "dwtdctsvd", "rivagan"]
    })

    decoder_arch: str = field(default="resnet50", metadata={
        "help": "name of the generation algorithm to use",
        "choices": ["stable-diffusion-1", "stable-diffusion-2"]  # currently only stable diffusion2 is supported
    })
