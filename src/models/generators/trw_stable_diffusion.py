from typing import Tuple

import torch
from diffusers import DPMSolverMultistepScheduler

from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.models.generators.inverse_stable_diffusion import InversableStableDiffusionPipeline
from src.models.generators.stable_diffusion import StableDiffusion
from src.utils.optim_utils import get_watermarking_mask, inject_watermark


class TRWStableDiffusion(StableDiffusion):
    """
    (Wen et al. 2023), paper: https://arxiv.org/abs/2305.20030

    This class modifies the generation process of the generator by injecting noise into its Fourier space.
    """

    def __init__(self, model_args: ModelArgs, env_args: EnvArgs = None):
        super().__init__(model_args, env_args)
        self.pipe = None

    def load(self) -> 'StableDiffusion':
        """
        Loads a stable diffusion pipeline
        """
        scheduler = DPMSolverMultistepScheduler.from_pretrained(self.model_args.model_ckpt,
                                                                subfolder='scheduler')
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            self.model_args.model_ckpt,
            scheduler=scheduler,
        )
        pipe.requires_safety_checker = False
        pipe.set_progress_bar_config(disable=True)
        pipe.to(self.env_args.device)
        self.pipe = pipe
        return self

    @torch.no_grad()
    def generate(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate images.
        """
        kwargs['num_images'] = kwargs.setdefault('num_images',
                                                 len(kwargs['latents']) if 'latents' in kwargs else kwargs[
                                                     'num_images'])
        kwargs['num_inference_steps'] = kwargs.setdefault('num_inference_steps', self.model_args.scheduler_timesteps)
        kwargs['guidance_scale'] = kwargs.setdefault('guidance_scale', self.model_args.guidance_scale)
        kwargs['prompt'] = kwargs.setdefault('prompt', [""] )
        kwargs['latents'] = self.pipe.get_random_latents(batch_size=kwargs['num_images']*len(kwargs['prompt'])).to(self.env_args.device)

        del kwargs['num_images']  # not accepted by pipeline
        kwargs['output_type'] = kwargs.setdefault('output_type', 'tensor')

        dtype = kwargs['latents'].dtype
        if self.wm_key is not None:  # generate with watermark
            watermarking_mask = get_watermarking_mask(kwargs['latents'], self.wm_key.wm_key_args, self.env_args.device)
            msg = self.wm_key.get_message()  #
            msg = msg.repeat([len(watermarking_mask), 1, 1, 1]).to(self.env_args.device)
            kwargs['latents'] = inject_watermark(kwargs['latents'], watermarking_mask, msg, self.wm_key.wm_key_args)
            kwargs['latents'] = kwargs['latents'].to(dtype)

        out = self.pipe(*args, **kwargs)
        return out.init_latents, out.images.float()

    def backward_diffusion(self, *args, **kwargs):
        """
        Generate images.
        """
        return self.pipe.backward_diffusion(*args, **kwargs)

    def get_text_embedding(self, *args, **kwargs):
        """
        Generate images.
        """
        return self.pipe.get_text_embedding(*args, **kwargs)

    def get_random_latents(self, *args, **kwargs):
        """
        Generate images.
        """
        return self.pipe.get_random_latents(*args, **kwargs)
