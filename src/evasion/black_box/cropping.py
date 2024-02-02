import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.arguments.evasion_args import CroppingEvasionArgs
from src.datasets.image_folder import ImageFolderDataset
from src.evasion.evasion_attack import BlackBoxEvasionAttack
from src.utils.utils import plot_images


class CroppingAttack(BlackBoxEvasionAttack):
    """
    Idea: Apply a blurring filter to the images and return the perturbed images
    """

    def attack(self, *args, **kwargs):
        """
        Apply a blurring filter to the images
        """
        self.attack_args: CroppingEvasionArgs
        src_dataset = ImageFolderDataset(self.attack_args.root_folder)  # the path to the source images
        data_loader = DataLoader(src_dataset, shuffle=False, batch_size=self.env_args.batch_size)
        scale = (self.attack_args.crop_scale, self.attack_args.crop_scale)
        ratio = (self.attack_args.crop_ratio, self.attack_args.crop_ratio)
        before_images = []
        cropped_images = []
        file_names = []
        for data, paths in tqdm(data_loader, disable=not self.attack_args.verbose_attack,
                                desc=f"Cropping {len(src_dataset)} images"):
            file_names += [os.path.basename(path) for path in paths]
            data = data.to(self.env_args.device)
            before_images.append(data)
            tensor = transforms.RandomResizedCrop(data.shape[-2:], scale=scale, ratio=ratio)(data)
            cropped_images.append(tensor)
        before_images = torch.cat(before_images)
        cropped_images = torch.cat(cropped_images)

        # logging
        if self.attack_args.verbose_attack:
            plot_images(torch.cat([before_images[-4:], cropped_images[-4:]], 0), n_row=4,
                        title="Cropping Before/After")
        self.save_images(cropped_images, names=file_names)
