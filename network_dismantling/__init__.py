import importlib
import logging
import pkgutil
from pathlib import Path
from typing import Union, List

import pandas as pd

from network_dismantling.common.data_structures import product_dict

try:
    from network_dismantling.common.dismantlers import dismantler_wrapper
except ImportError:
    dismantler_wrapper = None

dismantling_methods = {}

logger = logging.getLogger(__name__)


def setdefaultattr(obj, name, value):
    try:
        return getattr(obj, name)
    except AttributeError:
        setattr(obj, name, value)
    return value


class DismantlingMethod:
    key: str = None

    name: str = None
    short_name: str = None

    _depends_on: str = None

    doi: str = None
    citation: str = None
    description: str = None
    authors: Union[str, List[str]] = None

    function = None
    dynamic = None

    display_name: str = None
    short_display_name: str = None

    plot_color: str = None
    plot_marker: str = None

    # reinsertion: ReinsertionSupport = None
    includes_reinsertion: bool = False
    optional_reinsertion: bool = False
    reinsertion_function = None
    reinsertion_display_name = None
    reinsertion_short_display_name = None

    license_file: Path = None

    # return_type: ReturnTypes = None

    source: str = None

    def __init__(self,

                 **kwargs
                 ):

        super().__init__()

        for key, value in kwargs.items():
            # setdefaultattr(self, key, value)
            setattr(self, key, value)

        self.key = self.function.__name__

        if self.short_name is None:
            raise RuntimeError(f"Short name not defined for {self.key}")
        if not isinstance(self.short_name, str):
            raise RuntimeError(f"Short name not a string for {self.key}")


    def __call__(self, *args, **kwargs):
        output = self.function(*args, **kwargs)

        output["static"] = not self.dynamic
        output["heuristic"] = self.key

        return output

    def _format_output(self, output):

        if isinstance(output, dict):
            import pandas as pd

            output = pd.DataFrame(output)

        return output

    def _format_input(self, input: pd.DataFrame):
        return input

    def _filter_input(self, input: pd.DataFrame):
        return input

    def handle_parameters(self, **kwargs):
        return product_dict(kwargs)

    @property
    def depends_on(self):
        if self._depends_on is None:
            return None

        return dismantling_methods[self._depends_on]

    @depends_on.setter
    def depends_on(self, value):
        if value is None:
            self._depends_on = None
        else:

            if isinstance(value, str):
                self._depends_on = value
            elif isinstance(value, DismantlingMethod):
                self._depends_on = value.key
            else:
                self._depends_on = value.__name__


__all__ = []

for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    if module_name.startswith("_"):
        continue

    if module_name.endswith(".python_interface"):
        # print("Importing", module_name)
        _module = None
        try:
            _module = importlib.import_module(module_name)
        except Exception as e:
            logger.warning(f"Exception: {e}\n", exc_info=True)

            try:
                _module = loader.find_module(module_name).load_module(module_name)
            except Exception as e:
                logger.warning(f"Exception: {e}\n", exc_info=True)

        if _module is None:
            # print("Error importing:", module_name, e)
            logger.warning(f"Error importing {module_name.replace('.python_interface', '')}")

            continue
        else:
            # __alldict__[module_name] = _module
            __all__.append(module_name)
            globals()[module_name] = _module

__alldict__ = {k: globals()[k] for k in __all__}

# Unified interface exports
try:
    from .unified_interface import (
        dismantle,
        register_method,
        list_methods,
    )
    __all__.extend(["dismantle", "register_method", "list_methods"])
except ImportError as _e:
    logger.warning(f"Unified interface not available: {_e}")
