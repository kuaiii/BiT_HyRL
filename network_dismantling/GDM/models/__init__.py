
from network_dismantling.GDM.models.GAT import GAT_Model

import importlib
import pkgutil

from network_dismantling.GDM.models.base import BaseModel

for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    if module_name.startswith("_"):
        continue

    _module = None
    try:
        _module = importlib.import_module(f"GDM.models.{module_name}")
    except Exception as e:
        # print(f"Error loading sub-module {module_name} ({e})")

        try:
            _module = loader.find_module(f"GDM.models.{module_name}").load_module(f"GDM.models.{module_name}")
        except Exception as e:
            # print(f"Error loading sub-module {module_name} ({e})")
            pass

    if _module is None:
        print(f"Error loading sub-module {module_name}")
        pass
    else:
        globals()[module_name] = _module

models_mapping = dict((cls.__name__, cls) for cls in BaseModel.__subclasses__())

__all__ = list(models_mapping.items())
__alldict__ = models_mapping
