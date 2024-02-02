from src.arguments.env_args import EnvArgs
from src.arguments.quality_args import QualityArgs
from src.datasets.image_folder import ImageFolderDataset


class BaseMetric:

    def __init__(self, quality_args: QualityArgs, env_args: EnvArgs = None):
        self.quality_args = quality_args
        self.env_args = env_args

    @staticmethod
    def load_dataset(path, transform=None):
        """
        Load the underlying dataset.
        """
        return ImageFolderDataset(path, transform)

    def measure(self) -> float:
        raise NotImplementedError()

    def get_image_loader(self, folder_path, transform=None):
        """
        Create a DataLoader for image data.

        Args:
            folder_path (str): Path to the image folder.

        Returns:
            DataLoader: DataLoader for the images.
        """
        dataset = self.load_dataset(folder_path, transform)
        data_loader = self.env_args.make_data_loader(dataset, shuffle=False)
        return data_loader
