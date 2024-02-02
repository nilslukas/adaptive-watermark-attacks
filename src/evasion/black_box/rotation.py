import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.arguments.evasion_args import RotationEvasionArgs
from src.datasets.image_folder import ImageFolderDataset
from src.evasion.evasion_attack import BlackBoxEvasionAttack
from src.utils.utils import plot_images


class RotationAttack(BlackBoxEvasionAttack):
    """
    Idea: Apply a blurring filter to the images and return the perturbed images
    """

    def attack(self, *args, **kwargs):
        """
        Apply a blurring filter to the images
        """
        self.attack_args: RotationEvasionArgs
        src_dataset = ImageFolderDataset(self.attack_args.root_folder)  # the path to the source images
        data_loader = DataLoader(src_dataset, shuffle=False, batch_size=self.env_args.batch_size)
        before_images = []
        rotated_images = []
        file_names = []
        for data, paths in tqdm(data_loader, disable=not self.attack_args.verbose_attack,
                                desc=f"Rotating {len(src_dataset)} images"):
            file_names += [os.path.basename(path) for path in paths]
            data = data.to(self.env_args.device)
            before_images.append(data)
            image = transforms.RandomRotation((self.attack_args.rot_degree, self.attack_args.rot_degree))(data)
            rotated_images.append(image)
        before_images = torch.cat(before_images)
        rotated_images = torch.cat(rotated_images)
        # logging
        if self.attack_args.verbose_attack:
            plot_images(torch.cat([before_images[-4:], rotated_images[-4:]], 0), n_row=4,
                        title="Rotation Before/After")
        self.save_images(rotated_images, names=file_names)
