from dataclasses import dataclass, field

import yaml

from src.arguments.adaptive_diffuser_args import AdaptiveDiffuserArgs
from src.arguments.detect_args import DetectArgs
from src.arguments.embed_args import EmbedArgs
from src.arguments.env_args import EnvArgs
from src.arguments.evasion_args import EvasionArgs
from src.arguments.generate_image_args import GenerateImageArgs
from src.arguments.model_args import ModelArgs
from src.arguments.quality_args import QualityArgs
from src.arguments.wm_key_args import WatermarkingKeyArgs


@dataclass
class ConfigArgs:
    config_path: str = field(default=None, metadata={
        "help": "path to the yaml configuration file (*.yml)"
    })

    def exists(self):
        return self.config_path is not None

    args_to_config = {  # specify these keys in the *.yml file
        WatermarkingKeyArgs.CONFIG_KEY: WatermarkingKeyArgs,
        EnvArgs.CONFIG_KEY: EnvArgs,
        ModelArgs.CONFIG_KEY: ModelArgs,
        EmbedArgs.CONFIG_KEY: EmbedArgs,
        GenerateImageArgs.CONFIG_KEY: GenerateImageArgs,
        DetectArgs.CONFIG_KEY: DetectArgs,
        EvasionArgs.CONFIG_KEY: EvasionArgs,
        QualityArgs.CONFIG_KEY: QualityArgs,
        AdaptiveDiffuserArgs.CONFIG_KEY: AdaptiveDiffuserArgs
    }

    def get_embed_args(self) -> EmbedArgs:
        return self.loaded_configs.setdefault(EmbedArgs.CONFIG_KEY, EmbedArgs())

    def get_quality_args(self) -> QualityArgs:
        return self.loaded_configs.setdefault(QualityArgs.CONFIG_KEY, QualityArgs())

    def get_evasion_args(self) -> EvasionArgs:
        return self.loaded_configs.setdefault(EvasionArgs.CONFIG_KEY, EvasionArgs())

    def get_watermarking_key_args(self) -> WatermarkingKeyArgs:
        return self.loaded_configs.setdefault(WatermarkingKeyArgs.CONFIG_KEY, WatermarkingKeyArgs())

    def get_adaptive_diffuser_args(self) -> AdaptiveDiffuserArgs:
        return self.loaded_configs.setdefault(AdaptiveDiffuserArgs.CONFIG_KEY, AdaptiveDiffuserArgs())

    def get_detect_args(self) -> DetectArgs:
        return self.loaded_configs.setdefault(DetectArgs.CONFIG_KEY, DetectArgs())

    def get_generate_image_args(self) -> GenerateImageArgs:
        return self.loaded_configs.setdefault(GenerateImageArgs.CONFIG_KEY, GenerateImageArgs())

    def get_env_args(self) -> EnvArgs:
        return self.loaded_configs.setdefault(EnvArgs.CONFIG_KEY, EnvArgs())

    def get_model_args(self) -> ModelArgs:
        return self.loaded_configs.setdefault(ModelArgs.CONFIG_KEY, ModelArgs())

    def __post_init__(self):
        """
        Load from config file to dataclass
        """
        if self.config_path is None:
            print("> No config file specified. Using default values.")
            return
        self.loaded_configs = {}

        with open(self.config_path, "r") as f:
            data = yaml.safe_load(f)
        self.keys = list(data.keys())  # all keys specified in the yaml

        for entry in data.keys():
            cls = self.args_to_config[entry]
            values = {}
            if hasattr(cls, "from_checkpoint"):  # load from a local or remote checkpoint
                values = cls.from_checkpoint(**values)
            values.update(data[entry])  # yaml always overrides everything
            if hasattr(cls, "load_specific_class"):  # composability pattern
                cls = cls.load_specific_class(**values)
            self.loaded_configs[entry] = cls(**values)
