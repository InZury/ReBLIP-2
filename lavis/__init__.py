# Some portions of this file were inspired by code from [LAVIS].
# The original code is distributed under the BSD 3-Clause License, with the following copyright notice:
# Copyright (c) [2023], [salesforce.com, inc.]

import os
import sys

from omegaconf import OmegaConf
from lavis.utils.registry import registry
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.tasks import *

root_dir = os.path.dirname(os.path.abspath(__file__))
default_config = OmegaConf.load(os.path.join(root_dir, "config/default.yaml"))

registry.register_path("library_root", root_dir)
repo_root = os.path.join(root_dir, "..")
registry.register_path("repo_root", repo_root)
cache_root = os.path.join(repo_root, default_config.env.cache_root)
registry.register_path("cache_root", cache_root)

registry.register("MAX_INT", sys.maxsize)
registry.register("SPLIT_NAMES", ["train", "eval", "test"])
