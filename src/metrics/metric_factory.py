from src.arguments.env_args import EnvArgs
from src.arguments.quality_args import QualityArgs
from src.metrics.clipscore_metric import CLIPScoreMetric
from src.metrics.fid_metric import FIDMetric
from src.metrics.lpips_metric import LPIPSMetric
from src.utils.highlited_print import bcolors


class MetricFactory:

    @staticmethod
    def from_quality_args(quality_args: QualityArgs, env_args: EnvArgs = None):
        """
        Instantiate a quality evaluator
        """
        print(f"> Loading {bcolors.OKBLUE}{quality_args.metric}{bcolors.ENDC} metric")
        if quality_args.metric == "fid":
            return FIDMetric(quality_args, env_args)
        elif quality_args.metric == "clipscore":
            return CLIPScoreMetric(quality_args, env_args)
        elif quality_args.metric == "lpips":
            return LPIPSMetric(quality_args, env_args)

        raise ValueError(f"Unknown metric {quality_args.metric}")
