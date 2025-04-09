# Some portions of this file were inspired by code from [LAVIS].
# The original code is distributed under the BSD 3-Clause License, with the following copyright notice:
# Copyright (c) [2023], [salesforce.com, inc.]

from omegaconf import OmegaConf


class BaseProcessor:
    def __init__(self):
        self.transform = lambda x: x

        return

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, config=None):
        return cls()

    def build(self, **kwargs):
        config = OmegaConf.create(kwargs)

        return self.from_config(config)
