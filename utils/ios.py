import yaml
from munch import Munch


def load_config(fname):
    with open(fname, "r") as f:
        config = yaml.safe_load(f)
    config = Munch.fromDict(config)
    return config