import os
from io import BytesIO

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.arguments.evasion_args import JpegCompressionEvasionArgs
from src.datasets.image_folder import ImageFolderDataset
from src.evasion.evasion_attack import BlackBoxEvasionAttack
from src.utils.utils import plot_images


class JPEGCompressionAttack(BlackBoxEvasionAttack):
    """ Idea: Compress images to the images and return the perturbed images """

    def attack(self, *args, **kwargs):
        """
        Compress the images using JPEG and then decompress them to simulate an attack
        """
        src_dataset = ImageFolderDataset(self.attack_args.root_folder)  # the path to the source images
        data_loader = DataLoader(src_dataset, shuffle=False, batch_size=self.env_args.batch_size)
        self.attack_args: JpegCompressionEvasionArgs
        ctr = 0
        before_images = torch.empty((len(src_dataset), *src_dataset[0][0].shape))
        compressed_images = torch.empty((len(src_dataset), *src_dataset[0][0].shape))
        file_names = []
        for data, paths in tqdm(data_loader, disable=not self.attack_args.verbose_attack,
                                desc=f"JPEGing {len(src_dataset)} images"):
            file_names += [os.path.basename(path) for path in paths]
            for i in range(len(data)):
                before_images[ctr] = data[i]
                # Save the current image as a JPEG in a BytesIO object
                buffered = BytesIO()
                image = transforms.ToPILImage()(data[i]).convert("RGB")
                image.save(buffered, format='JPEG', quality=self.attack_args.jpeg_quality)

                # Load the image back from the BytesIO object
                image = Image.open(buffered)
                tensor = transforms.ToTensor()(image)

                compressed_images[ctr] = tensor
                ctr += 1

        # logging
        if self.attack_args.verbose_attack:
            plot_images(torch.cat([before_images[-4:], compressed_images[-4:]], 0), n_row=4,
                        title="JPEG Compression Before/After")
        self.save_images(compressed_images, names=file_names)
