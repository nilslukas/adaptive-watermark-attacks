import os
from typing import List

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from src.arguments.env_args import EnvArgs
from src.models.generators.image_generator import ImageGenerator
from src.utils.highlited_print import bcolors
from src.watermarking_key.wm_key import WatermarkingKey


class DiverseMessagesDataset(Dataset):
    """ A dataset for a two-level structure.
        <root>
            <class_label> # <- should correspond to a binary message
                <img 1>
                <img 2>
            <class_label>
                <img 1>

        returns img, class_label (parsed as a multi-class label)
    """

    def __init__(self, surrogate_generator: ImageGenerator,
                 watermarking_key: WatermarkingKey,
                 surrogate_decoder_args: None,
                 test_mode=False,
                 env_args: EnvArgs = None):
        assert watermarking_key is not None, "Need to specify a watermarking key. "

        self.root = surrogate_decoder_args.train_root
        self.env_args = EnvArgs() if env_args is None else env_args
        self.surrogate_decoder_args = surrogate_decoder_args
        print(
            f"> Loading {bcolors.OKBLUE}DiverseMessageDataset{bcolors.ENDC} from '{bcolors.OKGREEN}{self.root}{bcolors.ENDC}' (Test={test_mode})")

        # make sure we have enough images.
        self._generate_diverse_dataset(surrogate_generator, watermarking_key)

        # load all images.
        self.classes = list(sorted(self.discover_valid_classes()))[:self.surrogate_decoder_args.num_target_classes]

        self.image_files, self.image_labels = [], []
        for label, folder in enumerate(self.classes):
            path = os.path.join(self.root, folder)
            all_files = os.listdir(path)
            cutoff = int(len(all_files) * self.surrogate_decoder_args.test_split)
            all_files = all_files[:cutoff] if test_mode else all_files[cutoff:]
            for file in all_files:
                if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif')):
                    self.image_files += [os.path.join(path, file)]
                    self.image_labels += [label]

    def discover_valid_classes(self) -> List[str]:
        """
        Discover all folders that have sufficiently many images in them.
        """
        if not os.path.exists(self.root):
            return []
        all_classes = [class_label for class_label in os.listdir(self.root)]
        valid_classes = []
        for class_label in tqdm(all_classes, desc="Discover valid classes", disable=True):
            path = os.path.join(self.root, class_label)
            valid_files = [file for file in os.listdir(path) if
                           file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif'))]
            if len(valid_files) >= self.surrogate_decoder_args.num_images_per_class:
                valid_classes += [class_label]
        return valid_classes

    @torch.no_grad()
    def _generate_diverse_dataset(self,
                                  surrogate_generator: ImageGenerator,
                                  wm_key: WatermarkingKey):
        """
        Generate a diverse message dataset on the fly.
        """
        existing_classes: List[str] = self.discover_valid_classes()
        classes_to_generate = max(0, self.surrogate_decoder_args.num_target_classes - len(existing_classes))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # generate the
        for _ in tqdm(range(classes_to_generate), desc="Generate diverse message", disable=classes_to_generate == 0):
            wm_key.set_message(wm_key.sample_message(1)[0])  # set a random message
            msg_identifier = torch.randint(0, 99 ** 9, (1,)).item()

            os.makedirs(os.path.join(self.root, str(msg_identifier)), exist_ok=True)
            ctr = 0
            for _ in range(
                    int(np.ceil(self.surrogate_decoder_args.num_images_per_class / self.env_args.eval_batch_size))):
                seed = torch.randint(0, 99 ** 9, (1,)).item()
                _, x = surrogate_generator.generate(num_images=self.env_args.eval_batch_size, seed=seed)

                for x_i in x:
                    x_i = transforms.ToPILImage()(x_i)
                    x_i.save(os.path.join(self.root, str(msg_identifier), f"{ctr}.png"))
                    ctr += 1
        print(f"> Done generating all images!")

    def num_classes(self) -> int:
        return len(self.classes)

    def __len__(self):
        # Return the total number of image files in the dataset
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx])

        # Apply the transform to the image, if specified
        if self.transform:
            image = self.transform(image)

        # Return the image and the file path
        return image, self.image_labels[idx]
