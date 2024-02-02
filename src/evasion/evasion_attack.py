import os
from typing import List

import torch
from torch.utils.data import Dataset
from torchvision.utils import save_image
from tqdm import tqdm

from src.arguments.env_args import EnvArgs
from src.arguments.evasion_args import EvasionArgs
from src.utils.highlited_print import bcolors


class EvasionAttack:
    """ The basic attack class to evade a watermark """

    def __init__(self, attack_args: EvasionArgs, env_args: EnvArgs = None):
        self.attack_args = attack_args
        self.env_args = env_args

    def save_images(self, images: torch.Tensor):
        """ Save images to the output folder """
        raise NotImplementedError

    def attack(self, *args, **kwargs):
        """ Remove the watermark.  """
        raise NotImplementedError


class BlackBoxEvasionAttack(EvasionAttack):
    """ The basic attack class to evade a watermark """

    def __init__(self, attack_args: EvasionArgs, env_args: EnvArgs = None):
        super().__init__(attack_args, env_args)

    def load_image_dataset(self) -> Dataset:
        """ Load the image dataset from filepath and return torch.Dataset """
        raise NotImplementedError

    def save_images(self, images: torch.Tensor, names: List[str] = None):
        """ Save images to the output folder """
        output_folder = self.attack_args.output_folder
        os.makedirs(output_folder, exist_ok=True)
        for ctr, image in enumerate(tqdm(images, desc=f"Saving images to disk")):
            name = names[ctr] if names is not None else f"image_{ctr}.png"
            filename = os.path.join(output_folder, name)
            save_image(image.cpu(), filename)  # Save the image as PNG
        print(f"> Saved {images.shape} images to '{bcolors.OKGREEN}{os.path.abspath(output_folder)}{bcolors.ENDC}'.")

    def attack(self, *args, **kwargs) -> dict:
        """ Remove the watermark given a folder containing images.
         Returns a dict with output information. """
        raise NotImplementedError
