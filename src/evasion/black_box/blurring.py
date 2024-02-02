import os

import torch
from PIL import ImageFilter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.arguments.evasion_args import BlurringEvasionArgs
from src.datasets.image_folder import ImageFolderDataset
from src.evasion.evasion_attack import BlackBoxEvasionAttack
from src.utils.utils import plot_images


class BlurringAttack(BlackBoxEvasionAttack):
    """
    Idea: Apply a blurring filter to the images and return the perturbed images
    """

    def attack(self, *args, **kwargs):
        """
        Apply a blurring filter to the images
        """
        self.attack_args: BlurringEvasionArgs
        src_dataset = ImageFolderDataset(self.attack_args.root_folder)  # the path to the source images
        data_loader = DataLoader(src_dataset, shuffle=False, drop_last=False, batch_size=self.env_args.batch_size)

        ctr = 0
        before_images = torch.empty((len(src_dataset), *src_dataset[0][0].shape))
        blurred_images = torch.empty((len(src_dataset), *src_dataset[0][0].shape))
        file_names = []
        for data, paths in tqdm(data_loader, disable=not self.attack_args.verbose_attack,
                                desc=f"Blurring {len(src_dataset)} images"):
            file_names += [os.path.basename(path) for path in paths]
            for i in range(len(data)):
                # Convert the tensor image to a PIL image
                image = transforms.ToPILImage()(data[i]).convert("RGB")

                # Apply the Gaussian blur filter with a specified radius
                image = image.filter(ImageFilter.GaussianBlur(self.attack_args.blur_radius))

                # Convert the PIL image back to a tensor
                tensor = transforms.ToTensor()(image)

                blurred_images[ctr] = tensor
                before_images[ctr] = data[i]
                ctr += 1

        # logging
        if self.attack_args.verbose_attack:
            plot_images(torch.cat([before_images[-4:], blurred_images[-4:]], 0), n_row=4, title="Blurring Before/After")
        self.save_images(blurred_images, names=file_names)
