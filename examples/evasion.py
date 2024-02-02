import os

import transformers

from src.arguments.config_args import ConfigArgs
from src.arguments.env_args import EnvArgs
from src.arguments.evasion_args import EvasionArgs
from src.evasion.attack_factory import AttackFactory
from src.evasion.evasion_attack import EvasionAttack
from src.utils.utils import set_random_seed


def parse_args():
    parser = transformers.HfArgumentParser((EvasionArgs,  # arguments for the evasion attacks.
                                            EnvArgs,  # environment arguments
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()


def evasion_attack(evasion_args: EvasionArgs,  # points to images and has params for attack
                   env_args: EnvArgs,
                   config_args: ConfigArgs):
    """
    This script implements a set of attacks.

    Input:
    (1) A folder of (watermarked) images.

    Output:
    (1) A folder of images after using an evasion attack.
    """
    if config_args.exists():
        evasion_args = config_args.get_evasion_args()
        env_args = config_args.get_env_args()
    set_random_seed(env_args.seed)
    attack: EvasionAttack = AttackFactory.from_evasion_args(evasion_args=evasion_args, env_args=env_args)
    attack.attack()  # run the attack. all parameters are specified in evasion args.


if __name__ == "__main__":
    evasion_attack(*parse_args())
