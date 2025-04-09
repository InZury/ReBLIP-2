# Some portions of this file were inspired by code from [LAVIS].
# The original code is distributed under the BSD 3-Clause License, with the following copyright notice:
# Copyright (c) [2023], [salesforce.com, inc.]

import json

from typing import Dict
from omegaconf import OmegaConf
from logger import logger
from registry import registry


class Config:
    def __init__(self, args):
        self.config = {}
        self.args = args

        registry.register("configuration", self)

        user_config = self.build_opt_list(self.args.options)
        config = OmegaConf.load(self.args.config_path)
        runner_config = self.build_runner_config(config)
        model_config = self.build_model_config(config, **user_config)
        dataset_config = self.build_dataset_config(config)

        self.config = OmegaConf.merge(runner_config, model_config, dataset_config, user_config)

    @staticmethod
    def validate_runner_config(runner_config):
        runner_config_validator = create_runner_config_validator()
        runner_config_validator.validate(runner_config)

    def build_opt_list(self, opts):
        opts_dot_list = self.convert_to_dot_list(opts)

        return OmegaConf.from_dotlist(opts_dot_list)

    @staticmethod
    def build_runner_config(config):
        return {"run": config.run}

    @staticmethod
    def build_model_config(config, **kwargs):
        model = config.get("model", None)
        assert model is not None, "Missing model configuration file."

        model_class = registry.get_model_class(model.arch)
        assert model_class is not None, f"Model '{model.arch}' has not been registered."

        model_type = kwargs.get("model.model_type", None)

        if not model_type:
            model_type = model.get("model_type", None)

        assert model_type is not None, "Missing model_type."

        model_config_path = model_class.default_config_path(model_type=model_type)
        model_config = OmegaConf.create()
        model_config = OmegaConf.merge(
            model_config, OmegaConf.load(model_config_path), {"model": config["model"]}
        )

        return model_config

    @staticmethod
    def build_dataset_config(config):
        datasets = config.get("datasets", None)

        if datasets is None:
            raise KeyError("Expecting \"datasets\" as the root key for dataset configuration.")

        dataset_config = OmegaConf.create()

        for dataset_name in datasets:
            builder_class = registry.get_builder_class(dataset_name)
            dataset_config_type = datasets[dataset_name].get("type", "default")
            dataset_config_path = builder_class.default_config_path(type=dataset_config_type)

            dataset_config = OmegaConf.merge(
                dataset_config, OmegaConf.load(dataset_config_path),
                {"datasets": {dataset_name: config["datasets"][dataset_name]}}
            )

        return dataset_config

    @staticmethod
    def convert_to_dot_list(opts):
        if opts is None:
            opts = []
        if len(opts) == 0:
            return opts
        if opts[0].find('=') != -1:
            return opts

        return [f"{opt}={value}" for opt, value in zip(opts[0::2], opts[1::2])]

    def get_config(self):
        return self.config

    @property
    def run_config(self):
        return self.config.run

    @property
    def dataset_config(self):
        return self.config.datasets

    @property
    def model_config(self):
        return self.config.model

    def pretty_print(self):
        logger.info(
            f"\n=====  Running Parameters    =====\n"
            f"{self.convert_node_to_json(self.config.run)}"
        )

        logger.info("\n======  Dataset Attributes  ======")
        datasets = self.config.datasets

        for dataset in datasets:
            if dataset in self.config.datasets:
                logger.info(
                    f"\n======== {dataset} =======\n"
                    f"{self.convert_node_to_json(self.config.datasets[dataset])}"
                )
            else:
                logger.warning(f"No dataset named \"{dataset}\" in config. -- Skipping -- ")

        logger.info(
            f"\n======  Model Attributes  ======\n"
            f"{self.convert_node_to_json(self.config.model)}"
        )

    @staticmethod
    def convert_node_to_json(node):
        container = OmegaConf.to_container(node, resolve=True)
        return json.dumps(container, indent=4, sort_keys=True)

    def to_dict(self):
        return OmegaConf.to_container(self.config)


class ConfigValidator:
    class Argument:
        def __init__(self, name, choices=None, type_arg=None, help_arg=None):
            self.name = name
            self.validation = None
            self.choices = choices
            self.type = type_arg
            self.help = help_arg

        def __str__(self):
            string = f"{self.name}={self.validation}"

            if self.type is not None:
                string += f", ({self.type})"
            if self.choices is not None:
                string += f", choices: {self.choices}"
            if self.help is not None:
                string += f", ({self.help})"

            return string

    def __init__(self, description):
        self.description = description
        self.arguments = dict()
        self.parsed_args = None

    def __getitem__(self, key):
        assert self.parsed_args is not None, "No arguments parsed yet."

        return self.parsed_args[key]

    def __str__(self) -> str:
        return self.format_help()

    def add_argument(self, *args, **kwargs):
        self.arguments[args[0]] = self.Argument(*args, **kwargs)

    def validate(self, config=None):
        for key, value in config.items():
            assert (
                key in self.arguments
            ), f"{key} is not a validation arguments. Support arguments are {self.format_arguments()}."

            if self.arguments[key].type is not None:
                try:
                    self.arguments[key].validation = self.arguments[key].type(value)
                except ValueError:
                    raise ValueError(f"{key} is not a valid {self.arguments[key].type}")

            if self.arguments[key].choices is not None:
                assert (
                    value in self.arguments[key].choices
                ), f"{key} must be one of {self.arguments[key].chices}."

        return config

    def format_arguments(self):
        return str([f"{key}" for key in sorted(self.arguments.keys())])

    def format_help(self):
        return f"{self.description}, available arguments: {self.format_arguments()}"

    def print_help(self):
        print(self.format_help())


def node_to_dict(node):
    return OmegaConf.to_container(node)


def create_runner_config_validator():
    validator = ConfigValidator(description="Runner configurations")

    validator.add_argument(
        args="runner", type=str, choices=["runner_base", "runner_iter"],
        help="""Runner to use.
                The "runner_base" uses epoch-based training while iter-based runner runs based on iters.
                Default: runner_base"""
    )

    validator.add_argument(
        args="train_dataset_ratios", type=Dict[str, float],
        help="""Ratios of training dataset.
                This is used in iteration-based runner.
                Do not support for epoch-based runner because how to define an epoch becomes tricky.
                Default: None"""
    )

    validator.add_argument(
        args="max_iters", type=float,
        help="Maximum number of iterations to run."
    )

    validator.add_argument(
        args="max_epoch", type=int,
        help="Maximum number of epoches to run."
    )

    validator.add_argument(
        args="iters_per_inner_epoch", type=float,
        help="""Number of iterations per inner epoch.
                This is required when runner is runner_iter."""
    )

    validator.add_argument(
        args="lr_schedule", type=str, choices=registry.list_lr_schedulers(),
        help=f"Learning rate scheduler to use, from {registry.list_lr_schedulers()}"
    )

    validator.add_argument(
        args="task", type=str, choices=registry.list_tasks(),
        help=f"Task to use, from {registry.list_tasks()}."
    )

    validator.add_argument(
        args="init_lr", type=float,
        help="""Initial learning rate.
                This will be the learning rate after warmup and before decay."""
    )

    validator.add_argument(
        args="min_lr", type=float,
        help="Minimum learning rate (after decay)."
    )

    validator.add_argument(
        args="warmup_lr", type=float,
        help="Starting learning rate for warmup."
    )

    validator.add_argument(
        args="lr_decay_rate", type=float,
        help="""Learning rate decay rate.
                Required if using a decaying learning rate scheduler."""
    )

    validator.add_argument(
        args="weight_decay", type=float,
        help="Weight decay rate."
    )

    validator.add_argument(
        args="batch_size_train", type=int,
        help="Training batch size."
    )

    validator.add_argument(
        args="num_workers",
        help="Number of workers for data loading."
    )

    validator.add_argument(
        args="warmup_steps", type=int,
        help="""Number of warmup steps. 
                Required if a warmup schedule is used."""
    )

    validator.add_argument(
        args="seed", type=int,
        help="Random seed."
    )

    validator.add_argument(
        args="output_dir", type=str,
        help="Output directory to save checkpoints and logs."
    )

    validator.add_argument(
        args="evaluate",
        help="""Whether to only evaluate the model.
                If true, training will not be performed."""
    )

    validator.add_argument(
        args="train_splits", type=list,
        help="Splits to use for training."
    )

    validator.add_argument(
        args="validation_splits", type=list,
        help="""Splits to use for validation.
                If not provided, will skip the validation."""
    )

    validator.add_argument(
        args="test_splits", type=list,
        help="""Splits to use for testing.
                If not provided, will skip the testing."""
    )

    validator.add_argument(
        args="accumulate_grad_iters", type=int,
        help="Number of iterations to accumulate gradient for."
    )

    validator.add_argument(
        args="device", type=str, choices=["cpu", "cuda"],
        help="Device to use. Support \"cuda\" or \"cpu\" as for now."
    )

    validator.add_argument(
        args="world_size", type=int,
        help="Number of processes participating in the job."
    )

    validator.add_argument(args="dist_url", type=str)
    validator.add_argument(args="distributed", type=bool)

    validator.add_argument(
        args="use_dist_eval_sampler", type=bool,
        help="Whether to use distributed sampler during evaluation or not."
    )

    validator.add_argument(
        args="max_len", type=int,
        help="Maximal length of text output."
    )

    validator.add_argument(
        args="min_len", type=int,
        help="Minimal length of text output."
    )

    validator.add_argument(
        args="num_beams", type=int,
        help="Number of beams used for beam search."
    )

    validator.add_argument(
        args="num_answer_candidates", type=int,
        help="For ALBEF and BLIP, these models first rank answers according to likelihood to select answer candidates."
    )

    validator.add_argument(
        args="inference_method", type=str, choices=["generate", "rank"],
        help="""Inference method to use for question answering. 
                If rank, requires a answer list."""
    )

    validator.add_argument(
        args="k_test", type=int,
        help="Number of top k most similar samples from ITC/VTC selection to be tested."
    )

    return validator
