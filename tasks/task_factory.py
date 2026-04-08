# tasks/task_factory.py
from tasks.definitions.easy import EasyTask
from tasks.definitions.medium import MediumTask
from tasks.definitions.hard import HardTask


class TaskFactory:

    _registry = {
        "easy":   EasyTask,
        "medium": MediumTask,
        "hard":   HardTask,
    }

    @classmethod
    def create(cls, level: str):
        if level not in cls._registry:
            raise ValueError(
                f"Unknown task level '{level}'. "
                f"Choose from: {list(cls._registry.keys())}"
            )
        return cls._registry[level]()

    @classmethod
    def available_levels(cls):
        return list(cls._registry.keys())