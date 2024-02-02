import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.arguments.evasion_args import QuantizationEvasionArgs
from src.datasets.image_folder import ImageFolderDataset
from src.evasion.evasion_attack import BlackBoxEvasionAttack
from src.utils.utils import plot_images


class QuantizationAttack(BlackBoxEvasionAttack):
    """
    Idea: Apply a uniform quantization to the images and return the perturbed images
    """

    def attack(self, *args, **kwargs):
        """
        Apply a uniform quantization to the images
        """
        src_dataset = ImageFolderDataset(self.attack_args.root_folder)  # the path to the source images
        data_loader = DataLoader(src_dataset, shuffle=False, batch_size=self.env_args.batch_size)
        self.attack_args: QuantizationEvasionArgs
        ctr = 0
        before_images = torch.empty((len(src_dataset), *src_dataset[0][0].shape))
        quantized_images = torch.empty((len(src_dataset), *src_dataset[0][0].shape))
        file_names = []
        for data, paths in tqdm(data_loader, disable=not self.attack_args.verbose_attack,
                                desc=f"Quantizing {len(src_dataset)} images"):
            file_names += [os.path.basename(path) for path in paths]
            data = data.to(self.env_args.device)
            # Quantize the images
            before_images[ctr:ctr + len(data)] = data
            quantized_data = torch.floor(data * self.attack_args.num_levels) / self.attack_args.num_levels

            quantized_images[ctr:ctr + len(data)] = quantized_data
            ctr += len(data)

        # logging
        if self.attack_args.verbose_attack:
            plot_images(torch.cat([before_images[-4:], quantized_images[-4:]], 0), n_row=4,
                        title="Quantization Before/After")
        self.save_images(quantized_images, names=file_names)
