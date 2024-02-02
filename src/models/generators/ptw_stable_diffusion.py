from copy import deepcopy
from typing import Tuple, List, Optional, Union, Dict, Any

import torch

from src.model_converter.mappers.layer_wm_mapper import WMMapperGroup
from src.models.generators.stable_diffusion import StableDiffusion, rescale_noise_cfg


class PTWStableDiffusion(StableDiffusion):
    """
    A stable diffusion pipeline that is suitable for PTW key generation.
    The most important change is that we can activate/deactivate mappers during generation.
    """

    def generate(self,
                 w: Optional[torch.FloatTensor] = None,
                 num_images: Optional[int] = 1,
                 use_gradients_after: int = None,
                 prompt: Union[str, List[str]] = "",
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 guidance_scale: float = 0.0,
                 negative_prompt: Optional[Union[str, List[str]]] = None,
                 eta: float = 0.0,
                 generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                 prompt_embeds: Optional[torch.FloatTensor] = None,
                 negative_prompt_embeds: Optional[torch.FloatTensor] = None,
                 output_type: Optional[str] = "pt",
                 callback_steps: int = 1,
                 cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                 guidance_rescale: float = 0,
                 verbose=False,
                 mapper: WMMapperGroup = None,
                 **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a watermarked image
        """
        scheduler = deepcopy(self.pipe.scheduler)
        pipe = self.pipe.to(self.env_args.device)
        pipe.set_progress_bar_config(disable=not verbose)
        num_inference_steps = len(pipe.scheduler.timesteps)
        # 0. Default height and width to unet
        height = height or pipe.unet.config.sample_size * pipe.vae_scale_factor
        width = width or pipe.unet.config.sample_size * pipe.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        pipe.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = pipe._execution_device
        # pipe.scheduler.set_timesteps(50, device=device)
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )

        prompt_embeds = pipe._encode_prompt(
            prompt,
            device,
            num_images,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare timesteps
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = pipe.unet.config.in_channels
        latents = pipe.prepare_latents(
            batch_size * num_images,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            w,
        )
        noise = latents / pipe.scheduler.init_noise_sigma

        # 6. Prepare extra step kwargs
        extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)
        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
        decorator = torch.no_grad

        with pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                mapper.activate_mappers(False) if mapper is not None else None

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
                if use_gradients_after and i in use_gradients_after:
                    decorator = torch.enable_grad
                    mapper.activate_mappers(True) if mapper is not None else None

                with decorator():
                    # predict the noise residual
                    noise_pred = pipe.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if do_classifier_free_guidance and guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                    # compute the previous noisy sample x_t -> x_t-1

                    latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                        progress_bar.update()

        if not output_type == "latent":
            image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents

        image = pipe.image_processor.postprocess(image, output_type=output_type)

        # Offload last model to CPU
        if hasattr(pipe, "final_offload_hook") and pipe.final_offload_hook is not None:
            pipe.final_offload_hook.offload()
        mapper.activate_mappers(True) if mapper is not None else None
        self.pipe.scheduler = scheduler
        return noise, image
