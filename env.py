import os
import shutil


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_env(config, config_name, path):
    os.makedirs(path, exist_ok=True)
    shutil.copyfile(config, os.path.join(path, config_name))
