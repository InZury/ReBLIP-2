# Some portions of this file were inspired by code from [LAVIS].
# The original code is distributed under the BSD 3-Clause License, with the following copyright notice:
# Copyright (c) [2023], [salesforce.com, inc.]

import os
import torch
import torch.distributed as dist

from lavis.utils.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from lavis.utils.logger import logger, MetricLogger, SmoothedValue
from lavis.utils.registry import registry
from lavis.datasets.data_utils import prepare_sample


class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()
        _ = kwargs
        self.instance_id_key = "instance_id"

    @classmethod
    def setup_task(cls, **kwargs):
        _ = kwargs

        return cls()

    @staticmethod
    def build_model(config):
        model_config = config.model_config
        model_class = registry.get_model_class(model_config.arch)

        return model_class.from_config(model_config)

    @staticmethod
    def build_datasets(config):
        datasets = dict()
        datasets_config = config.datasets_config

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            datasets_config = datasets_config[name]
            builder = registry.get_builder_class(name)(datasets_config)
            dataset = builder.build_datasets()
            datasets[name] = dataset

        return datasets

    @staticmethod
    def train_step(model, samples):
        output = model(samples)
        loss_dict = {}

        for key, value in output.items():
            if "loss" in key:
                loss_dict[key] = value

        return output["loss"], loss_dict

    def valid_step(self, model, samples):
        raise NotImplementedError

    def before_training(self, model, dataset, **kwargs):
        _ = kwargs
        model.before_training(dataset=dataset, task_type=type(self))

    def before_evaluation(self, model, dataset, **kwargs):
        _ = kwargs
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError

    def evaluation(self, model, data_loader, enabled=True):
        metric = MetricLogger(delimiter="  ")
        header = "Evaluation"
        frequency = 10
        results = []

        for samples in metric.log_every(data_loader, frequency, header):
            samples = prepare_sample(samples, enabled=enabled)
            eval_output = self.valid_step(model=model, samples=samples)

            results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    def train_epoch(self, epoch, model, data_loader, optimizer, lr_scheduler,
                    scaler=None, enabled=False, frequency=50, accumulated_grad_iters=1):
        return self.train_inner_loop(
            epoch=epoch, iters_per_epoch=len(data_loader),
            model=model, data_loader=data_loader,
            optimizer=optimizer, lr_scheduler=lr_scheduler,
            scaler=scaler, frequency=frequency,
            enabled=enabled, accumulated_grad_iters=accumulated_grad_iters
        )

    def train_iters(self, epoch, start_iters, iters_per_inner_epoch, model, data_loader, optimizer, lr_scheduler,
                    scaler=None, enabled=False, frequency=50, accumulated_grad_iters=1):
        return self.train_inner_loop(
            epoch=epoch, iters_per_epoch=iters_per_inner_epoch,
            model=model, data_loader=data_loader,
            optimizer=optimizer, lr_scheduler=lr_scheduler,
            scaler=scaler, start_iters=start_iters, frequency=frequency,
            enabled=enabled, accumulated_grad_iters=accumulated_grad_iters
        )

    def train_inner_loop(self, epoch, iters_per_epoch, model, data_loader, optimizer, lr_scheduler,
                         scaler=None, start_iters=None, enabled=False, frequency=50, accumulated_grad_iters=1):
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            data_loader = iter(data_loader)

        metric = MetricLogger(delimiter="  ")
        metric.add_meter("lr", SmoothedValue(window_size=1, formating="{value:.6f}"))
        metric.add_meter("loss", SmoothedValue(window_size=1, formating="{value:.4f}"))

        logger.info(f"Start training epoch {epoch}, {iters_per_epoch} iters per inner epoch.")

        header = f"Train: data epoch: [{epoch}]"

        if start_iters is None:
            inner_epoch = epoch
        else:
            inner_epoch = start_iters // iters_per_epoch
            header = f"{header}; inner epoch [{inner_epoch}]"

        for iteration in metric.log_every(range(iters_per_epoch), frequency, header):
            if iteration >= iters_per_epoch:
                break

            samples = next(data_loader)
            samples = prepare_sample(samples, enabled=enabled)

            if not isinstance(samples, dict):
                samples = {"is_empty": True}

            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": iteration
                }
            )

            lr_scheduler.step(current_epoch=inner_epoch, current_step=iteration)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, loss_dict = self.train_step(model=model, samples=samples)
                loss /= accumulated_grad_iters

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (iteration + 1) % accumulated_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()

            metric.update(**loss_dict)
            metric.update(lr=optimizer.param_groups[0]["lr"])

        metric.synchronize_between_processes()
        logger.info(f"Averaged stats: {metric.global_avg()}")

        return {
            key: f"{meter.global_avg:.3f}"
            for key, meter in metric.meters.items()
        }

    @staticmethod
    def save_result(results, result_dir, file_name, remove_duplicate=''):
        import json

        result_file = os.path.join(result_dir, f"{file_name}_rank{get_rank()}.json")
        final_result_file = os.path.join(result_dir, f"{file_name}.json")

        json.dump(results, open(result_file, 'w'))

        if is_dist_avail_and_initialized():
            dist.barrier()
        if is_main_process():
            logger.warning(f"rank {get_rank()} starts merging results.")

            results = []

            for rank in range(get_world_size()):
                result_file = os.path.join(result_dir, f"{file_name}_rank{rank}.json")
                result = json.load(open(result_file, 'r'))
                results += result

            if remove_duplicate:
                result_new = []
                id_list = []

                for result in results:
                    if result[remove_duplicate] not in id_list:
                        id_list.append(result[remove_duplicate])
                        result_new.append(result)

                results = result_new

            json.dump(results, open(final_result_file, 'w'))
            print(f"result file saved to {final_result_file}")

        return final_result_file
