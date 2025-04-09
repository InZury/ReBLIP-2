# Some portions of this file were inspired by code from [LAVIS].
# The original code is distributed under the BSD 3-Clause License, with the following copyright notice:
# Copyright (c) [2023], [salesforce.com, inc.]

import logging
import inspect
import datetime
import time
import torch
import torch.distributed as dist

from collections import defaultdict, deque


class SmoothedValue(object):
    def __init__(self, window_size=20, formating=None):
        if formating is None:
            formating = "{median:.4f} ({global_avg:.4f})"

        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.format = formating

    def update(self, value, count=1):
        self.deque.append(value)
        self.count += count
        self.total = value * count

    def synchronize_between_processes(self):
        import lavis.utils.dist_utils as dist_utils

        if not dist_utils.is_dist_avail_and_initialized():
            return

        tensor = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(tensor)
        tensor = tensor.tolist()

        self.count = int(tensor[0])
        self.total = tensor[1]

    @property
    def median(self):
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self):
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.format.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()

            assert isinstance(value, (float, int))

            self.meters[key].update(value)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]

        raise AttributeError(f"'{type(self).__name__}' object has not attribute '{attr}'.")

    def __str__(self):
        loss_str = []

        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter}")

        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []

        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter.global_avg:.4f}")

        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, frequency, header=None):
        iteration = 0

        if not header:
            header = ""

        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(formating="{avg:.4f}")
        data_time = SmoothedValue(formating="{avg:.4f}")
        digits = len(str(len(iterable)))
        mb = 1024.0 * 1024.0

        for task in iterable:
            data_time.update(time.time() - end)
            yield task
            iter_time.update(time.time() - end)

            if torch.cuda.is_available():
                memory = f"max memory: {(torch.cuda.max_memory_allocated() / mb):.0f}"
            else:
                memory = ''

            if iteration % frequency == 0 or iteration == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - iteration)

                print(f"{header} [{iteration:{digits}d} /{len(iterable)}] "
                      f"eta: {datetime.timedelta(seconds=int(eta_seconds))} " 
                      f"{self} " 
                      f"time: {iter_time} data: {data_time} "
                      f"{memory}")

            iteration += 1
            end = time.time()

        total_time = time.time() - start_time
        total = (f"{header} Total time: {datetime.timedelta(seconds=int(total_time))} "
                 f"({(total_time / len(iterable)):.4f} s / it)")

        print(f"{'-' * len(total)}\n"
              f"{total}\n"
              f"{'-' * len(total)}")


class LoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        frame = inspect.stack()[-1]
        module = inspect.getmodule(frame[0])
        file_name = module.__file__.split("\\")[-1] if module and hasattr(module, "__file__") else ""

        self.logger.name = file_name

        return msg, kwargs


init_logger = logging.getLogger("Init")
formatter = logging.Formatter("%(asctime)s|%(levelname)s|%(name)s| - %(message)s", "%Y-%m-%d %H:%M:%S")
handler = logging.StreamHandler()
handler.setFormatter(formatter)

if not init_logger.handlers:
    init_logger.addHandler(handler)
    init_logger.setLevel(logging.INFO)

logger = LoggerAdapter(init_logger, {})
