import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.arguments.evasion_args import ImageJitteringEvasionArgs
from src.datasets.image_folder import ImageFolderDataset
from src.evasion.evasion_attack import BlackBoxEvasionAttack
from src.utils.utils import plot_images


class ImageJitteringAttack(BlackBoxEvasionAttack):
    """ Apply random jitter to the images and return the perturbed images """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def attack(self, *args, **kwargs):
        """
        Randomly jitter the images to simulate an attack
        """
        self.attack_args: ImageJitteringEvasionArgs
        src_dataset = ImageFolderDataset(self.attack_args.root_folder)
        data_loader = DataLoader(src_dataset, shuffle=False, batch_size=self.env_args.batch_size)
        strength = self.attack_args.jitter_strength
        jitter = transforms.ColorJitter(brightness=strength)
        before_images = []
        jittered_images = []
        file_names = []
        # Apply random jitter to the image
        for data, paths in tqdm(data_loader, disable=not self.attack_args.verbose_attack,
                                desc=f"Jittering {len(src_dataset)} images"):
            data = data.to(self.env_args.device)
            file_names += [os.path.basename(path) for path in paths]
            before_images.append(data)
            jittered_batch = jitter(data)
            jittered_images.append(jittered_batch)

        before_images = torch.cat(before_images)
        jittered_images = torch.cat(jittered_images)

        if self.attack_args.verbose_attack:
            plot_images(torch.cat([before_images[-4:], jittered_images[-4:]], 0), n_row=4,
                        title="Jittering Before/After")
        self.save_images(jittered_images, names=file_names)
