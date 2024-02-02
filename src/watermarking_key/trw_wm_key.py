import os

import scipy
import torch
from diffusers import DPMSolverMultistepScheduler

from src.arguments.wm_key_args import TRWWatermarkingKeyArgs, WatermarkingKeyArgs
from src.models.generators.image_generator import ImageGenerator
from src.models.generators.inverse_stable_diffusion import InversableStableDiffusionPipeline
from src.utils.highlited_print import bcolors
from src.utils.optim_utils import get_watermarking_pattern, get_watermarking_mask
from src.utils.web import download_and_unzip
from src.watermarking_key.wm_key import WatermarkingKey


class TRWWatermarkingKey(WatermarkingKey):
    """
    The TRW watermarking key. Can be used to generate watermarked images and to extract the watermark.
    """

    def __init__(self, wm_key_args: TRWWatermarkingKeyArgs, **kwargs):
        super().__init__(wm_key_args=wm_key_args, **kwargs)
        self.mask = None
        self.text_embedding = None
        self.wm_key_args: TRWWatermarkingKeyArgs
        self.pipe = None

    def sample_message(self, n: int) -> torch.Tensor:
        """
        Sample {n} random binary messages.
        """
        self.wm_key_args: TRWWatermarkingKeyArgs
        messages = get_watermarking_pattern(self.pipe, self.wm_key_args, self.env_args.device, shape=(n, 4, 64, 64))
        return messages

    def embed(self, generator: ImageGenerator, message: torch.Tensor):
        """
        Embeds a message into a generator's parameters.
        """
        pass  # No embedding required. Just make sure to load a TRW compatible generator.

    def extract_message_with_gradients(self, x: torch.Tensor):
        img_w = x.to(self.text_embedding.dtype).to(self.env_args.device)
        image_latents_w = self.pipe.get_image_latents(img_w, sample=False)

        reversed_latents_w = self.pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=self.text_embedding[:1].repeat_interleave(len(image_latents_w), dim=0),
            guidance_scale=1,
            num_inference_steps=self.wm_key_args.reversal_inference_steps,
        )
        reversed_latents_w = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))
        return reversed_latents_w

    def verify_message_with_gradients(self, msg_pred, msg_true):
        if len(msg_true.shape) != len(msg_pred.shape):
            msg_true = msg_true.unsqueeze(0)
        if msg_true.shape[0] != msg_pred.shape[0]:
            msg_true = msg_true.repeat_interleave(msg_pred.shape[0], dim=0)
        mask = get_watermarking_mask(msg_pred, self.wm_key_args, self.env_args.device)

        distance = -torch.linalg.norm(msg_pred[mask]+msg_true[mask])
        return distance  # summing real and imaginary parts

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts an embedded_message from images.
        @param x should be in [n, c, h, w]
        """
        img_w = x.to(self.text_embedding.dtype).to(self.env_args.device)
        image_latents_w = self.pipe.get_image_latents(img_w, sample=False)

        reversed_latents_w = self.pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=self.text_embedding[:1].repeat_interleave(len(image_latents_w), dim=0),
            guidance_scale=1,
            num_inference_steps=self.wm_key_args.reversal_inference_steps,
        )

        if 'complex' in self.wm_key_args.w_measurement:
            reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))
        elif 'seed' in self.wm_key_args.w_measurement:
            reversed_latents_w_fft = reversed_latents_w
        else:
            raise NotImplementedError(f'w_measurement: {self.wm_key_args.w_measurement}')

        return reversed_latents_w_fft  # this is the message [b, 4, 64, 64,]

    @torch.no_grad()
    def verify(self, msg_pred: torch.Tensor, msg_true: torch.Tensor) -> dict:
        """
        Extracts an embedded_message from images and computes a p-value for the confidence that the embedded
        and extracted messages are matching.
        """
        results = {
            'p_values': [],
            'distances': []  # optional, if l1 distances are used. WARNING: Won't return an accuracy then
        }

        if len(msg_true.shape) != len(msg_pred.shape):
            msg_true = msg_true.unsqueeze(0)
        if msg_true.shape[0] != msg_pred.shape[0]:
            msg_true = msg_true.repeat_interleave(msg_pred.shape[0], dim=0)
        mask = get_watermarking_mask(msg_pred, self.wm_key_args, self.env_args.device)

        for ix in range(msg_pred.shape[0]):
            y = msg_pred[ix][mask[ix]]
            k_star = msg_true[ix][mask[ix]]
            M = y.shape[0]
            sigma_2 = torch.mean(torch.abs(y)**2)
            eta = torch.mean(torch.abs(k_star-y)**2) / sigma_2 * M
            lambda_ = torch.mean(torch.abs(k_star) ** 2) / sigma_2 * M
            p = scipy.stats.ncx2.cdf(eta.item(), M, lambda_.item())
            results['p_values'].append(p)
        results['accuracy'] = len([p for p in results['p_values'] if p < self.p_value_threshold]) / len(results['p_values'])

        return results

    def set_message(self, msg: torch.Tensor):
        real_part = msg.real.float()
        imaginary_part = msg.imag.float()
        self.wm_key_args.message = torch.stack([real_part, imaginary_part], 0)

    def get_message(self):
        return torch.complex(self.wm_key_args.message[0], self.wm_key_args.message[1]).to(
            self.env_args.device)

    def keygen(self, generator: ImageGenerator = None):
        """
        Generate a watermarking key given the generator.
        """
        assert generator is not None, "Generator must be provided. We need to know what model will be used. "

        model_args = generator.model_args
        self.wm_key_args.inversal_model = model_args.model_ckpt  # remember the model checkpoint

        scheduler = DPMSolverMultistepScheduler.from_pretrained(self.wm_key_args.inversal_model,
                                                                subfolder='scheduler')
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            self.wm_key_args.inversal_model,
            scheduler=scheduler,
        )
        pipe.requires_safety_checker = False
        pipe.to(self.env_args.device)

        # hardcode latent shape, because ALL stable diffusion <= 2.1 models have this latent shape
        pipe.set_progress_bar_config(disable=not self.env_args.verbose)
        self.set_message(self.sample_message(1)[0])
        self.pipe = pipe
        self.save(ckpt_fn=self.wm_key_args.key_ckpt)
        self.text_embedding = self.pipe.get_text_embedding('').repeat([self.env_args.batch_size, 1, 1])
        self.mask = get_watermarking_mask(self.pipe.get_random_latents(), self.wm_key_args, self.env_args.device)
        self.mask = self.mask.squeeze(0)
        return self

    def save(self, ckpt_fn: str = None) -> dict:
        """
        Save all necessary data to re-instantiate the key into a single '*.pt' file
        """
        state_dict = {
            WatermarkingKeyArgs.WM_KEY_ARGS_KEY: self.wm_key_args,
        }
        if ckpt_fn:
            torch.save(state_dict, ckpt_fn)
            print(f"> Saved TRW watermarking key to '{bcolors.OKGREEN}{os.path.abspath(ckpt_fn)}{bcolors.ENDC}'")
        return state_dict

    def load(self, ckpt=None):
        """
        Load all necessary data to re-instantiate the key.
        """
        ckpt = ckpt if ckpt is not None else torch.load(download_and_unzip(self.wm_key_args.key_ckpt), map_location='cpu')
        self.wm_key_args: TRWWatermarkingKeyArgs
        self.wm_key_args = ckpt[WatermarkingKeyArgs.WM_KEY_ARGS_KEY]
        print(f"> Loaded a TRW watermarking key!")

        scheduler = DPMSolverMultistepScheduler.from_pretrained(self.wm_key_args.inversal_model, subfolder='scheduler')
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            self.wm_key_args.inversal_model,
            scheduler=scheduler,
        )

        pipe.set_progress_bar_config(disable=not self.env_args.verbose)
        pipe.requires_safety_checker = False
        pipe.to(self.env_args.device)
        self.pipe = pipe
        self.text_embedding = self.pipe.get_text_embedding('').repeat([self.env_args.batch_size, 1, 1])
        self.mask = get_watermarking_mask(self.pipe.get_random_latents(), self.wm_key_args, self.env_args.device)
        self.mask = self.mask.squeeze(0)
        return self
