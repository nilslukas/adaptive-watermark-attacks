import json
import os

from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm


class PromptDataset(Dataset):
    """Load a dataset from an image folder.

    Parameters:
        root (string): Path to the root directory of the dataset.
    """

    def __init__(self, root):
        self.root = root
        self.dataset = []
        self.key = ""
        if "coco" in root:
            with open(root) as f:
                self.dataset = json.load(f)['annotations']
                self.key = 'caption'
        else:
            self.dataset = load_dataset(root)['test']
            self.key = 'Prompt'

    def __len__(self):
        # Return the total number of prompts in the dataset
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.dataset[idx][self.key]
        else:
            return [v[self.key] for v in self.dataset[idx]]

    def save(self, dir_path, suffix="_0"):
        os.makedirs(dir_path, exist_ok=True)
        for i in tqdm(range(len(self.dataset)), desc=f"saving prompt dataset to {dir_path}"):
            prompt = self.__getitem__(i)
            filename = f"image_{i}{suffix}.txt"
            filepath = os.path.join(dir_path, filename)
            with open(filepath, "w+") as f:
                f.write(prompt)
