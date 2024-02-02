import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.arguments.evasion_args import GaussianNoiseEvasionArgs
from src.datasets.image_folder import ImageFolderDataset
from src.evasion.evasion_attack import BlackBoxEvasionAttack
from src.utils.utils import plot_images


class GaussianNoiseAttack(BlackBoxEvasionAttack):
    """
    Add Gaussian noise to the images
    """

    def attack(self, *args, **kwargs):
        """ Add Gaussian Noise to the images """
        self.attack_args: GaussianNoiseEvasionArgs
        src_dataset = ImageFolderDataset(self.attack_args.root_folder)  # the path to the source images
        data_loader = DataLoader(src_dataset, shuffle=False, batch_size=self.env_args.batch_size)

        ctr = 0
        before_images = torch.empty((len(src_dataset), *src_dataset[0][0].shape))
        noised_images: torch.Tensor = torch.empty((len(src_dataset), *src_dataset[0][0].shape))
        file_names = []
        for data, paths in tqdm(data_loader, disable=not self.attack_args.verbose_attack,
                                desc=f"Noising {len(src_dataset)} images"):
            file_names += [os.path.basename(path) for path in paths]
            data = data.to(self.env_args.device)
            before_images[ctr:ctr + len(data)] = data
            noised_images[ctr:ctr + len(data)] = torch.clamp(
                data + self.attack_args.gaussian_noise_scale * torch.randn_like(data), 0, 1)
            ctr += len(data)

        # logging
        if self.attack_args.verbose_attack:
            plot_images(torch.cat([before_images[-4:], noised_images[-4:]], 0), n_row=4,
                        title="Gaussian Noise Before/After")
        self.save_images(noised_images, names=file_names)
