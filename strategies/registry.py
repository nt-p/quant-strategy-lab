"""Strategy auto-discovery registry.

Scans the strategies/ folder and returns one instance of every class
that inherits from StrategyBase (except StrategyBase itself).

Drop a new .py file in strategies/ → it appears in the dashboard.
No other changes needed.

Skipped files
-------------
- base.py      — contains the ABC itself
- registry.py  — this file
- __init__.py  — package marker
- __*.py       — any dunder-prefixed file
"""

import importlib
import logging
import pkgutil
from pathlib import Path

from .base import StrategyBase

logger = logging.getLogger(__name__)

_SKIP = {"base", "registry", "__init__"}


def discover_strategies() -> list[StrategyBase]:
    """Scan the strategies/ package and return one instance per StrategyBase subclass.

    Modules that raise ImportError (e.g. missing optional dependency) are
    logged as warnings and skipped rather than crashing the whole dashboard.

    Returns
    -------
    list[StrategyBase]
        Sorted by (category.value, name) so the sidebar groups stay stable.
    """
    strategies: list[StrategyBase] = []
    package_dir = Path(__file__).parent

    for _, module_name, _ in pkgutil.iter_modules([str(package_dir)]):
        # Skip internal/dunder modules
        if module_name in _SKIP or module_name.startswith("__"):
            continue

        try:
            module = importlib.import_module(f".{module_name}", package="strategies")
        except ImportError as exc:
            logger.warning("Skipping strategy module %r — import failed: %s", module_name, exc)
            continue
        except Exception as exc:
            logger.warning("Skipping strategy module %r — unexpected error: %s", module_name, exc)
            continue

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, StrategyBase)
                and attr is not StrategyBase
            ):
                strategies.append(attr())

    return sorted(strategies, key=lambda s: (s.category.value, s.name))
