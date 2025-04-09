# Some portions of this file were inspired by code from [LAVIS].
# The original code is distributed under the BSD 3-Clause License, with the following copyright notice:
# Copyright (c) [2023], [salesforce.com, inc.]

class Registry:
    mapping = {
        "builder_name_mapping": {},
        "task_name_mapping": {},
        "processor_name_mapping": {},
        "model_name_mapping": {},
        "lr_scheduler_name_mapping": {},
        "runner_name_mapping": {},
        "state": {},
        "paths": {}
    }

    @classmethod
    def register_builder(cls, name):
        def wrap(builder_cls):
            from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

            assert issubclass(
                builder_cls, BaseDatasetBuilder
            ), f"All builders must inherit BaseDatasetBuilder class, found {builder_cls}"

            if name in cls.mapping["builder_name_mapping"]:
                raise KeyError(f"Name {name} already registered for {cls.mapping['builder_name_mapping'][name]}.")

            cls.mapping["builder_name_mapping"][name] = builder_cls

            return builder_cls

        return wrap

    @classmethod
    def register_task(cls, name):
        def wrap(task_cls):
            from lavis.tasks.base_task import BaseTask

            assert issubclass(
                task_cls, BaseTask
            ), "All tasks must inherit BaseTask class"

            if name in cls.mapping["task_name_mapping"]:
                raise KeyError(
                    f"Name '{name}' already registered for {cls.mapping['task_name_mapping'][name]}."
                )

            cls.mapping["task_name_mapping"][name] = task_cls

            return task_cls

        return wrap

    @classmethod
    def register_model(cls, name):
        def wrap(model_cls):
            from lavis.models import BaseModel

            assert issubclass(
                model_cls, BaseModel
            ), "All models must inherit BaseModel class"

            if name in cls.mapping["model_name_mapping"]:
                raise KeyError(
                    f"Name '{name}' already registered fpr {cls.mapping['model_name_mapping'][name]}."
                )

            cls.mapping["model_name_mapping"][name] = model_cls

            return model_cls

        return wrap

    @classmethod
    def register_processor(cls, name):
        def wrap(processor_cls):
            from lavis.processors import BaseProcessor

            assert issubclass(
                processor_cls, BaseProcessor
            ), "All processors must inherit BaseProcessor class"

            if name in cls.mapping["processor_name_mapping"]:
                raise KeyError(
                    f"Name '{name}' already registered for {cls.mapping['processor_name_mapping'][name]}."
                )

            cls.mapping["processor_name_mapping"][name] = processor_cls

            return processor_cls

        return wrap

    @classmethod
    def register_lr_scheduler(cls, name):
        def wrap(lr_scheduler_cls):
            if name in cls.mapping["lr_scheduler_name_mapping"]:
                raise KeyError(
                    f"Name '{name}' already registered for {cls.mapping['lr_scheduler_name_mapping'][name]}."
                )

            cls.mapping["lr_scheduler_name_mapping"][name] = lr_scheduler_cls

            return lr_scheduler_cls

        return wrap

    @classmethod
    def register_runner(cls, name):
        def wrap(runner_cls):
            if name in cls.mapping["runner_name_mapping"]:
                raise KeyError(
                    f"Name '{name}' already registered for {cls.mapping['runner_name_mapping'][name]}."
                )

            cls.mapping["runner_name_mapping"][name] = runner_cls

            return runner_cls

        return wrap

    @classmethod
    def register_path(cls, name, path):
        assert isinstance(path, str), "All path must be str."

        if name in cls.mapping["paths"]:
            raise KeyError(f"Name '{name}' already registered.")

        cls.mapping["paths"][name] = path

    @classmethod
    def register(cls, name, obj):
        path = name.split('.')
        current = cls.mapping["state"]

        for part in path[:-1]:
            if part not in current:
                current[part] = {}

            current = current[part]

        current[path[-1]] = obj

    @classmethod
    def get_builder_class(cls, name):
        return cls.mapping["builder_name_mapping"].get(name, None)

    @classmethod
    def get_model_class(cls, name):
        return cls.mapping["model_name_mapping"].get(name, None)

    @classmethod
    def get_task_class(cls, name):
        return cls.mapping["task_name_mapping"].get(name, None)

    @classmethod
    def get_processor_class(cls, name):
        return cls.mapping["processor_name_mapping"].get(name, None)

    @classmethod
    def get_lr_scheduler_class(cls, name):
        return cls.mapping["lr_scheduler_name_mapping"].get(name, None)

    @classmethod
    def get_runner_class(cls, name):
        return cls.mapping["runner_name_mapping"].get(name, None)

    @classmethod
    def list_runners(cls):
        return sorted(cls.mapping["runner_name_mapping"].keys())

    @classmethod
    def list_models(cls):
        return sorted(cls.mapping["model_name_mapping"].keys())

    @classmethod
    def list_tasks(cls):
        return sorted(cls.mapping["task_name_mapping"].keys())

    @classmethod
    def list_processors(cls):
        return sorted(cls.mapping["processor_name_mapping"].keys())

    @classmethod
    def list_lr_schedulers(cls):
        return sorted(cls.mapping["lr_scheduler_name_mapping"].keys())

    @classmethod
    def list_datasets(cls):
        return sorted(cls.mapping["builder_name_mapping"].keys())

    @classmethod
    def get_path(cls, name):
        return cls.mapping["paths"].get(name, None)

    @classmethod
    def get(cls, name, default=None, no_warning=False):
        original_name = name
        name = name.split('.')
        value = cls.mapping["state"]

        for sub_name in name:
            value = value.get(sub_name, default)

            if value is default:
                break

        if "writer" in cls.mapping["state"] and value == default and not no_warning:
            cls.mapping["state"]["writer"].warning(
                f"Key {original_name} is not present in registry, returning default value of {default}."
            )

        return value

    @classmethod
    def unregister(cls, name):
        return cls.mapping["state"].pop(name, None)


registry = Registry()

if __name__ == "__main__":
    from lavis.models import load_model_and_preprocess

    model, vis = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True)

    print(vis)
