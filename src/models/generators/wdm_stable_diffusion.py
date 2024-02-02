from typing import Tuple, List, Optional, Union, Dict, Any

import torch

from src.models.generators.stable_diffusion import StableDiffusion
from src.watermarking_key.wdm_wm_key import WDMWatermarkingKey


class WDMStableDiffusion(StableDiffusion):
    """
    A stable diffusion pipeline that is suitable for WDM watermarking method.
    """

    def generate(self,
                 w: Optional[torch.FloatTensor] = None,
                 num_images: Optional[int] = 1,
                 use_gradients_after: int = None,
                 prompt: Union[str, List[str]] = "",
                 height: Optional[int] = 512,
                 width: Optional[int] = 512,
                 guidance_scale: float = 7.5,
                 negative_prompt: Optional[Union[str, List[str]]] = None,
                 eta: float = 0.0,
                 generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                 prompt_embeds: Optional[torch.FloatTensor] = None,
                 negative_prompt_embeds: Optional[torch.FloatTensor] = None,
                 output_type: Optional[str] = "pt",
                 callback_steps: int = 1,
                 cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                 guidance_rescale: float = 0.0,
                 verbose=False,
                 **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate images.
        """
        pipe = self.pipe.to(self.env_args.device)
        pipe.set_progress_bar_config(disable=not verbose)

        num_inference_steps = len(pipe.scheduler.timesteps)
        n_prompts = 1 if isinstance(prompt, str) else len(prompt)

        height = height or pipe.unet.config.sample_size * pipe.vae_scale_factor
        width = width or pipe.unet.config.sample_size * pipe.vae_scale_factor
        num_channels_latents = pipe.unet.in_channels

        size = n_prompts * num_images
        if w is None:
            w = pipe.prepare_latents(
                size,
                num_channels_latents,
                height,
                width,
                pipe.text_encoder.dtype,
                self.env_args.device,
                generator,
                None,
            )

        images = pipe(
            prompt,
            num_images_per_prompt=num_images,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            latents=w,
            output_type="pt",
        ).images

        with torch.no_grad():
            if self.wm_key is not None:  # apply post-processing
                self.wm_key: WDMWatermarkingKey
                self.wm_key.encoder = self.wm_key.encoder.eval().to(self.env_args.device)
                images = self.wm_key.encoder(images, message=self.wm_key.get_message())
        return w, images
