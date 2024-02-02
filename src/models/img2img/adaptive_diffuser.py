import os

import torch
from diffusers import AutoencoderTiny, AutoencoderKL, StableDiffusionImg2ImgPipeline, EulerDiscreteScheduler, \
    StableDiffusionImageVariationPipeline
from torch import nn

from src.arguments.adaptive_diffuser_args import AdaptiveDiffuserArgs
from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.models.autoencoders.convramp import ConvRamp
from src.models.autoencoders.skipencoder import SkipEncoderDecoder
from src.models.autoencoders.stega import StegaStampEncoder
from src.models.autoencoders.taesd import TAESD
from src.models.generators.stable_diffusion import StableDiffusion
from src.utils.highlited_print import bcolors
from src.utils.web import download_and_unzip


class AdaptiveDiffuser(nn.Module):
    ADAPTIVE_DIFF_KEY = "adaptive_diffuser_key"
    ADAPTIVE_DIFF_STATE_DICT_KEY = "adaptive_diffuser_state_key"

    def __init__(self, adaptive_diffuser_args: AdaptiveDiffuserArgs, env_args: EnvArgs = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptive_diffuser_args = adaptive_diffuser_args
        self.env_args = env_args if env_args is not None else EnvArgs()

        self.img2img_fwd = None
        self.img2img = None
        self.encoder = None
        self.decoder = None
        self._init()  # initialize the models

    def _init(self):
        """
        Initialize the models.
        """
        print(
            f"> Initializing adaptive diffuser with img2img model '{bcolors.OKGREEN}{self.adaptive_diffuser_args.img2img_ckpt}{bcolors.ENDC}'")
        if self.adaptive_diffuser_args.img2img_ckpt in ["stabilityai/sdxl-vae"]:
            img2img = AutoencoderKL.from_pretrained(self.adaptive_diffuser_args.img2img_ckpt)
            img2img_fwd = lambda x: img2img(x.to(img2img.dtype)).sample
        elif self.adaptive_diffuser_args.img2img_ckpt in ["madebyollin/taesd"]:
            img2img = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16)
            self.encoder = img2img.encoder
            self.decoder = img2img.decoder
            img2img_fwd = lambda x: img2img.decoder(img2img.encoder(x.to(img2img.dtype))).float()
        elif self.adaptive_diffuser_args.img2img_ckpt in ["surrogate_vae"]:
            generator = StableDiffusion(model_args=ModelArgs(model_name="stable-diffusion-1",
                                                             model_ckpt="CompVis/stable-diffusion-v1-1"),
                                        env_args=self.env_args)
            img2img = generator.load().pipe.vae
            img2img_fwd = lambda x: img2img(x.to(img2img.dtype)).sample
        elif self.adaptive_diffuser_args.img2img_ckpt in ["stega"]:
            img2img = StegaStampEncoder(resolution=512, fingerprint_size=16)
            img2img_fwd = lambda x: img2img(x, message=torch.zeros((len(x), 16)).to(x.device))
        elif self.adaptive_diffuser_args.img2img_ckpt in ['taesd']:
            img2img = TAESD()
            img2img.decoder.load_state_dict(torch.load(f"../pretrained_models/taesd_decoder.pth"))
            img2img.encoder.load_state_dict(torch.load(f"../pretrained_models/taesd_encoder.pth"))
            img2img_fwd = lambda x: img2img(x)
        else:
            raise ValueError(f"Unknown img2img_ckpt: {self.adaptive_diffuser_args.img2img_ckpt}")
        print(
            f"> {bcolors.OKBLUE}Img2Img model has {sum(p.numel() for p in img2img.parameters()) / 10 ** 6:.2f}M parameters {bcolors.ENDC}")

        self.img2img_fwd = img2img_fwd
        self.img2img = img2img

    def __call__(self, x: torch.Tensor):
        return self.img2img_fwd(x)

    def load(self, ckpt=None):
        """
        Load the state dict from a checkpoint file.
        """
        ckpt = ckpt if ckpt is not None else torch.load(download_and_unzip(self.adaptive_diffuser_args.save_path),
                                                        map_location='cpu')
        save_location = self.adaptive_diffuser_args.save_path
        print(f"> Loaded adaptive diffuser from '{bcolors.OKGREEN}{save_location}{bcolors.ENDC}'")
        print(ckpt.keys())
        self.adaptive_diffuser_args = ckpt[self.ADAPTIVE_DIFF_KEY]
        self._init()
        self.load_state_dict(ckpt[self.ADAPTIVE_DIFF_STATE_DICT_KEY])
        self.adaptive_diffuser_args.save_path = save_location
        return self


    def save(self, ckpt_fn: str = None) -> dict:
        """ Save the state dict. """

        state_dict = {
            self.ADAPTIVE_DIFF_KEY: self.adaptive_diffuser_args,
            self.ADAPTIVE_DIFF_STATE_DICT_KEY: self.state_dict()
        }
        if ckpt_fn is not None:
            torch.save(state_dict, ckpt_fn)
            print(f"> Saved adaptive diffuser to '{bcolors.OKGREEN}{os.path.abspath(ckpt_fn)}{bcolors.ENDC}'")
        return state_dict
