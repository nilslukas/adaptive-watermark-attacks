import os

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class ImageFolderDataset(Dataset):
    """Load a dataset from an image folder.

    Parameters:
        root (string): Path to the root directory of the dataset.
        transform (callable, optional): A function/transform to apply to the images.
    """

    def __init__(self, root, transform=None):
        self.root = root

        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        self.transform = transform

        # Initialize empty list to store image file paths
        self.image_files = []

        for image_file in sorted(os.listdir(self.root)):
            # Check if the file is an image
            if (image_file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif'))
                    and not image_file.startswith(".")):
                image_file_path = os.path.join(self.root, image_file)
                self.image_files.append(image_file_path)

    def __len__(self):
        # Return the total number of image files in the dataset
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load the image file at index `idx`
        image = Image.open(self.image_files[idx]).convert('RGB')

        # Apply the transform to the image, if specified
        if self.transform:
            image = self.transform(image)

        # Return the image and the file path
        return image, self.image_files[idx]
