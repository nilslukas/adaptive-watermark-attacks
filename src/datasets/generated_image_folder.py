import os

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.utils.highlited_print import bcolors


class GeneratedImageFolderDataset(Dataset):
    """
     Load a dataset from an image folder.

    """

    def __init__(self, root, model: torch.nn.Module, num_pre_generate: int = 100, batch_size: int = 1):
        super(GeneratedImageFolderDataset, self).__init__()

        self.root = root
        self.dir_path = os.path.join(self.root, f"batch_size_{batch_size}")
        self.num_pre_generate = num_pre_generate
        self.batch_size = batch_size
        # self.wm_key_args = wm_key_args if wm_key_args is not None else WatermarkingKeyArgs()
        # self.env_args = env_args if env_args is not None else EnvArgs()
        self.model = model

        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

        # Initialize empty lists to store image file paths and corresponding labels
        self.seeds = []
        self.available_batch_sizes = []
        for image_dir in os.listdir(self.root):
            if "batch_size_" in image_dir.lower():
                self.available_batch_sizes.append(int(image_dir[image_dir.rfind("batch_size_") + len("batch_size_"):]))
        self.size = 0
        # first = True
        for image_file in os.listdir(self.dir_path):
            # Check if the file is an image
            if image_file.lower().endswith(".pt"):
                image_file_path = os.path.join(self.dir_path, image_file)
                try:
                    checkpoint = torch.load(image_file_path, map_location='cpu')
                except:
                    print(
                        f"{bcolors.WARNING} [WARNING] {os.path.abspath(image_file_path)} is corrupted. Please remove it {bcolors.ENDC}")
                    continue
                assert image_file_path == os.path.join(self.dir_path, f"{checkpoint['seed']}.pt"), \
                    f"Make sure to store pre_generated images as *.pt where * is the seed associated with " \
                    f"(given seed: {checkpoint['seed']}, associated file path: {image_file_path}"
                assert checkpoint["size"] == self.batch_size, f"found file {image_file_path} with batch size " \
                                                              f"{checkpoint['size']} in folder" \
                                                              f"{self.dir_path} instead of the required batch size of " \
                                                              f"{self.batch_size}, please remove it"
                # if checkpoint["size"] < self.batch_size:
                #     if first:
                #         print(f"batch size found in pre generated batch is smaller ({checkpoint['size']}) than"
                #               f" requested batch size ({self.batch_size}), will generate the additionally needed "
                #               f"images")
                #         first = False
                #     self.increase_size(checkpoint, image_file_path, self.batch_size-checkpoint["size"])
                #     checkpoint = torch.load(image_file_path)
                # if checkpoint["size"] < self.max_batch_size:
                #     self.max_batch_size = checkpoint["size"]
                self.seeds.append(checkpoint["seed"])
                self.size += checkpoint["size"]
            if self.size > self.num_pre_generate > 0:
                break

        if self.size < self.num_pre_generate:
            print(f"> Pre-generating {self.num_pre_generate - self.size} images")
            print(f"Model_args: {self.model.model_args}")
            print(f"Batch Size: {batch_size}")
            self.pre_generate(self.num_pre_generate - self.size)

    def pre_generate(self, num_images: int):
        with torch.no_grad():
            num_batches = num_images // self.batch_size  # we use train batch size (usually smaller than eval batch size) so the same seed doesn't get repeated too much
            remainder = num_images % self.batch_size
            print()
            for _ in tqdm(range(num_batches), desc="Generating images"):
                seed = torch.randint(0, 99 ** 9, (1,)).item()
                file_path = os.path.join(self.dir_path, f"{seed}.pt")

                if not os.path.exists(file_path):
                    w, x = self.model.generate(num_images=self.batch_size, seed=seed)

                    torch.save({
                        "seed": seed,
                        "noise": w,
                        "image": x,
                        "size": self.batch_size
                    }, f=file_path)
                    self.seeds.append(seed)
                    self.size += self.batch_size

            # we also generate the remainder
            if remainder != 0:
                seed = torch.randint(0, 99 ** 9, (1,)).item()
                file_path = os.path.join(self.dir_path, f"{seed}.pt")
                while os.path.exists(file_path):
                    # make sure we find a seed that we didn't already compute
                    seed = torch.randint(0, 99 ** 9, (1,)).item()
                    file_path = os.path.join(self.dir_path, f"{seed}.pt")
                # if os.path.exists(file_path):
                #     checkpoint = torch.load(file_path)
                #     if checkpoint["size"] < self.batch_size:
                #         w, x = self.model.generate(num_images=(self.batch_size - checkpoint["size"]), seed=seed)
                #         torch.save({
                #             "seed": seed,
                #             "noise": torch.cat([checkpoint["noise"], w], dim=0),
                #             "image": torch.cat([checkpoint["image"], x], dim=0),
                #             "size": self.batch_size
                #         }, f=file_path)
                #         if seed not in self.seeds:
                #             self.seeds.append(seed)
                #             self.size += self.batch_size
                #         else:
                #             self.seeds.append(seed)
                #             self.size += self.batch_size-checkpoint["size"]
                #         return

                w, x = self.model.generate(num_images=self.batch_size, seed=seed)
                torch.save({
                    "seed": seed,
                    "noise": w,
                    "image": x,
                    "size": self.batch_size
                }, f=file_path)
                self.seeds.append(seed)
                self.size += self.batch_size

    def __getitem__(self, index):
        checkpoint = torch.load(os.path.join(self.dir_path, f"{self.seeds[index]}.pt"), map_location="cpu")
        assert checkpoint["size"] == self.batch_size, \
            f"saved seed batch of seed {self.seeds[index]} has a different size than the demanded batch size. " \
            f"Something must have gone wrong with the initialization of the dataset and/or the size increase."

        return checkpoint["seed"], checkpoint["noise"], checkpoint["image"]

    def __len__(self):
        # not returning the true size but instead the number of seeds we have generated to trick the dataloader
        return len(self.seeds)
