import os
from abc import ABC
from os.path import dirname
from typing import List

import scipy
import torch
import torchvision
import wandb
from math import sqrt
from scipy.stats import binom
from torch.utils import data
from torch.utils.data import DataLoader

from src.arguments.env_args import EnvArgs
from src.arguments.wm_key_args import WatermarkingKeyArgs, PTWWatermarkingKeyArgs
from src.datasets.generated_image_folder import GeneratedImageFolderDataset
from src.models.generators.image_generator import ImageGenerator
from src.utils.highlited_print import bcolors
from src.utils.smoothed_value import SmoothedValue
from src.utils.utils import plot_images


class WatermarkingKey:

    def __init__(self, wm_key_args: WatermarkingKeyArgs, env_args: EnvArgs = None):
        """
        The base class for a watermarking key. A key maps from 
        
        (i) [message, parameters] -> [watermarked parameters] (called 'embedding')
        (ii) [watermarked image] -> [message] (called 'extraction')
        
        """
        super().__init__()
        self.wm_key_args = wm_key_args
        self.env_args = EnvArgs() if env_args is None else env_args  # assume default env arguments if none are given.

    def keygen(self, generator=None):
        """ 
        Generate a watermarking key. Optionally accepts a generator (optional)
        """
        raise NotImplementedError

    def sample_message(self, n: int) -> List[torch.Tensor]:
        """
        Sample {n} random messages.
        """
        raise NotImplementedError

    def embed(self, generator: ImageGenerator, message: torch.Tensor):
        """
        Embeds a message into a generator's parameters.
        """
        raise NotImplementedError

    def extract(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extracts an embedded_message from images.
        """
        raise NotImplementedError

    @property
    def p_value_threshold(self):
        return 0.01  # lower p-value means the watermark is present

    def verify(self, x: torch.Tensor, message: torch.Tensor) -> dict:
        """
        Extracts an embedded_message from images and computes a p-value for the confidence. 
        """
        raise NotImplementedError

    def get_message(self):
        """
        Get the current watermarking message
        """
        return self.wm_key_args.message.float()

    def set_message(self, message):
        """
        Set the watermarking message that should be embedded.
        """
        self.wm_key_args.message = message

    # ------------------------------------------------------------

    def save(self, ckpt_fn: str = None) -> dict:
        """ 
        Saves a key to a single '*.pt' file. If no ckpt_fn is given, only returns the save dict.
        """
        raise NotImplementedError

    def load(self, ckpt=None):
        """ 
        Loads a key from a '*.pt' file. 
        """
        raise NotImplementedError

    def extract_message_with_gradients(self, x: torch.Tensor):
        raise NotImplementedError

    def verify_message_with_gradients(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        raise NotImplementedError


class MultiBitWatermarkingKey(WatermarkingKey, ABC):
    """
    A watermarking key that uses binary strings as messages.
    """

    def sample_message(self, n: int) -> torch.Tensor:
        """
        Sample {n} random binary messages.
        """
        self.wm_key_args: PTWWatermarkingKeyArgs
        return torch.randint(0, 2, (n, self.wm_key_args.bitlen)).float().to(self.env_args.device)

    def get_message(self):
        """
        Get the current watermarking message
        """
        return self.wm_key_args.message.to(self.env_args.device).float()

    @torch.no_grad()
    def verify(self, msg_pred: torch.Tensor, msg_true: torch.Tensor) -> dict:
        """
        Computes a dict with p-values to reject the null hypothesis.
        """
        data = {
            'p_values': [],
        }
        # print(f"True message: {msg_true}")
        msg_true = msg_true.to(msg_pred.device)

        if msg_true.shape[0] != msg_pred.shape[0]:
            msg_true = msg_true[:1].repeat_interleave(len(msg_pred), dim=0)

        for msg_pred_i, msg_true_i in zip(msg_pred, msg_true):
            matches = torch.sum(msg_pred_i == msg_true_i).item()
            n = len(msg_true_i)
            p_expected = 0.5
            if self.wm_key_args.double_tail:
                tail1_p_value = binom.cdf(matches, n, p_expected)
                tail2_p_value = 1 - binom.cdf(matches - 1, n, p_expected)
                double_tail_p_value = 2 * min(tail1_p_value, tail2_p_value)
                double_tail_p_value = min(double_tail_p_value, 1)  # Ensure p-value does not exceed 1
                data['p_values'].append(double_tail_p_value)
            else:
                p_value = 1 - binom.cdf(matches - 1, n, p_expected)
                data['p_values'].append(p_value)

        data['accuracy'] = len([x for x in data['p_values'] if x < self.p_value_threshold]) / len(data['p_values'])

        return data


class TrainableWatermarkingKey(WatermarkingKey):

    def __init__(self, wm_key_args: WatermarkingKeyArgs, env_args: EnvArgs = None):
        super().__init__(wm_key_args, env_args=env_args)
        self.data_loader = None

    @staticmethod
    def gradient_check(nn_module, name=""):
        """
        Checks if the given module's parameters have gradients attached to them
        """
        total_available = 0
        total_unavailable = 0
        data = []

        for param_name, param in nn_module.named_parameters():
            if param.grad is None:
                print("2", param_name)
                total_unavailable += param.numel()
            else:
                data += [param.grad.mean()]  # any operation on the gradient
                total_available += param.numel()

        if total_available == 0:
            print(
                f">{bcolors.FAIL} [ERROR] '{name}' has {total_unavailable / 10 ** 6:.3f}M parameters without gradients!{bcolors.ENDC}")
        elif total_available > 0 and total_unavailable != 0:
            print(
                f"> {bcolors.WARNING} [WARNING] '{name}' has {total_available / 10 ** 6:.3f}M parameters with gradients "
                f"and {total_unavailable} params without. Might be a bug, but does not have to be.{bcolors.ENDC}")
        else:
            print(
                f"> {bcolors.OKGREEN} [SUCCESS] '{name}' has {total_available / 10 ** 6:.3f}M parameters with gradients.{bcolors.ENDC}")

    def log_data(self, step, bit_acc: SmoothedValue, extracted_msg, msg, pbar, x, x_wm, loss_dict: dict, run=None):
        """
        Logs the data to the logging tool
        """
        self.wm_key_args: PTWWatermarkingKeyArgs

        print()
        print(f"----------------------------------------------------------------------------")
        print(
            f"> Step={step}, Bits={loss_dict['capacity']:.2f}, Bit_acc={bcolors.OKGREEN}{bit_acc.avg:.2f}%{bcolors.ENDC}, "
            f"Target Bits: {self.wm_key_args.bitlen}, Save Interval: {self.env_args.save_every}")

        print(f"----------------------------------------------------------------------------")
        print()

        pbar.reset(total=self.env_args.log_every)

        top = [x for x in x_wm[:3].cpu()]
        middle = [x for x in x[:3].cpu()]
        bottom = [x - y for x, y in zip(x_wm[:3].cpu(), x[:3].cpu())]
        plot_images(torch.stack(top + middle + bottom, 0), n_row=len(top),
                    title=f"step={step}, bits={loss_dict['capacity']:.2f}")
        if self.env_args.get_accelerator().is_local_main_process and run is not None:
            images = wandb.Image(
                torchvision.utils.make_grid(torch.stack(top + middle + bottom, 0),
                                            nrow=len(top), range=(-1, 1), scale_each=True,
                                            normalize=True),
                caption="Top: Watermarked, Middle: Original, Bottom: Diff"
            )
            wandb.log({"examples": images})

    def print_shield(self):
        """
        Print statistics about the training.
        """
        print()
        print(f"> --------------------------------------------- ")
        print(f"> Generating a {bcolors.OKGREEN}{str(self.__class__)}{bcolors.ENDC} watermarking key.")
        if self.wm_key_args.key_ckpt is not None:
            os.makedirs(dirname(self.wm_key_args.key_ckpt), exist_ok=True)
            print(
                f"> Checkpoint directory: '{bcolors.OKGREEN}{os.path.abspath(self.wm_key_args.key_ckpt)}{bcolors.ENDC}'")
        else:
            print(f"> {bcolors.WARNING}[WARNING] No checkpoint directory given. Will NOT save the key.{bcolors.ENDC}")
        print(f"> Bit length: {bcolors.OKBLUE}{self.wm_key_args.bitlen}{bcolors.ENDC}")
        print(f"> Logging: {bcolors.OKBLUE}{self.env_args.logging_tool}{bcolors.ENDC}")
        print(f"> --------------------------------------------- ")
        print()

    def get_data_loader(self, generator: ImageGenerator) -> data.DataLoader:
        """
        We load pre-generated images that have been generated with a generator.
        """
        self.wm_key_args: PTWWatermarkingKeyArgs
        data_path = os.path.join(self.wm_key_args.pre_generation_folder,
                                 generator.model_args.model_name,
                                 generator.model_args.model_ckpt,
                                 f"timesteps_{generator.model_args.scheduler_timesteps}")

        pre_generated_data = GeneratedImageFolderDataset(root=data_path, model=generator,
                                                         num_pre_generate=self.wm_key_args.num_pre_generate,
                                                         batch_size=self.env_args.batch_size)
        data_loader = DataLoader(pre_generated_data,
                                 batch_size=1,  # needs to be 1 because we fix the seed.
                                 shuffle=True,
                                 num_workers=self.env_args.num_workers)
        print(f"> Found {len(pre_generated_data)} pre-generated images in '{bcolors.OKGREEN}{data_path}{bcolors.ENDC}'")
        return data_loader

    def get_logger(self):
        run = None
        if self.env_args.logging_tool == "wandb" and self.env_args.get_accelerator().is_local_main_process:
            wandb.login()
            run = wandb.init(project="ptw_keygen", entity="diffusion-v2")
        return run

    def load_next_batch(self):
        """
        An infinite data loader.
        """
        while True:
            for seed, w, x in self.data_loader:
                yield seed.item(), w[0], x[0]
