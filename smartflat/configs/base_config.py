"""Base Config class."""

import inspect
import json
from copy import deepcopy
from typing import Any, Dict

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    
class BaseConfig:
    """Base config class.

    Implements shared parameters handling.

    To keep config from exploding, we only encode the first layer of nesting
    If we need sub-configs, those are free-form dicts and are passed as-is
    to the underlying builder

    Implements shared helper methods for persisting and loading configs.
    """
    
    
    
    def __init__(self):
        self.config_name = self.__class__.__name__
    #
    # Serialization Methods
    #
    def to_dict(self) -> Dict[str, Any]:
        """Export config to a python dict."""
        # List all public attributes, excluding methods and functions
        attributes = [
            name
            for name in dir(self)
            if not name.startswith("_") and not inspect.ismethod(getattr(self, name))
        ]

        return deepcopy({name: getattr(self, name) for name in attributes})

    def to_json(self, filename: str):
        """Export config to disk in json format."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4, sort_keys=True, cls=NumpyEncoder)
            


    @classmethod
    def from_dict(cls, config_dict):
        """Create a config object from a python dict."""
        config = cls()
        config_dict = deepcopy(config_dict)
        for key, value in config_dict.items():
            setattr(config, key, value)
        return config

    @classmethod
    def from_json(cls, filename):
        """Create a config object from a json file."""
        with open(filename, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def clone(self):
        """Clone config."""
        return self.__class__.from_dict(self.to_dict())

    def update(
        self,
        key,
        value,
    ):
        """This function will update values based on the nesting level of the parameter.

        key = "attr_name.key.subkey.subsubkey"
        value = 1

        We expect dotfile format. The first attribute needs to be an attribute.
        The remaining are keys in dicts

        config.attr_name[key][subkey][subsubkey] = 1
        """

        attr_path = key.split(".")

        if len(attr_path) == 1:
            # easy case, just set an attribute on the parent directly
            # e.g config.optimizer_name = "sgd"
            setattr(self, key, value)
        else:
            # we need to traverse the dict hierarchy.
            # e.g config.dataset_params.batch_size = 256.
            # if config does not have a `dataset_params` attribute, we need to create it
            if not hasattr(self, attr_path[0]):
                setattr(self, attr_path[0], {})
            attr_dict = getattr(self, attr_path[0])
            attr_path = attr_path[1:]

            for idx, subkey in enumerate(attr_path):
                if idx == len(attr_path) - 1:
                    attr_dict[subkey] = value
                    break

                if subkey not in attr_dict:
                    attr_dict[subkey] = {}

                attr_dict = attr_dict[subkey]

    def is_valid(self):
        """Check if the combination of hyperparameters is valid.
        Can be used to abort early when doing hyperparameter tuning.
        """
        return True
