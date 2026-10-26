"""Finding node classes, and merging the settings they contribute."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from dataclasses import dataclass, field
from collections.abc import Iterable

from .node import Node
from .settings import SettingSpec, SettingsRegistry


def tree_root() -> str:
    """Dotted name of the package that holds graph/, nodes/ and families/ ('' at top level)."""
    parts = (__package__ or "").split(".")
    return ".".join(parts[:-1])


def _qualify(root: str, name: str) -> str:
    return f"{root}.{name}" if root else name


@dataclass
class NodeRegistry:
    """Scans packages for concrete Node subclasses. ``class_name`` is the identity."""

    packages: tuple[str, ...] = ()
    classes: dict[str, type[Node]] = field(default_factory=dict)
    errors: list[dict] = field(default_factory=list)
    _loaded: bool = False

    @classmethod
    def default(cls) -> NodeRegistry:
        root = tree_root()
        names = [_qualify(root, "nodes")]
        try:
            families = importlib.import_module(_qualify(root, "families"))
            for info in pkgutil.iter_modules(families.__path__):
                if info.ispkg and info.name not in {"api", "general"}:
                    names.append(_qualify(root, f"families.{info.name}.nodes"))
        except ImportError:
            pass
        return cls(packages=tuple(names))

    def load(self) -> tuple[dict[str, type[Node]], list[dict]]:
        if not self._loaded:
            self.reload()
        return self.classes, self.errors

    def reload(self) -> None:
        self.classes, self.errors = {}, []
        for package_name in self.packages:
            try:
                package = importlib.import_module(package_name)
            except ModuleNotFoundError as exc:
                if exc.name == package_name:
                    continue  # a family without a nodes/ folder
                self.errors.append({"module": package_name, "error": f"{type(exc).__name__}: {exc}"})
                continue
            except Exception as exc:  # noqa: BLE001 -- a broken module must not hide the others
                self.errors.append({"module": package_name, "error": f"{type(exc).__name__}: {exc}"})
                continue
            modules = [package]
            if hasattr(package, "__path__"):
                for info in pkgutil.walk_packages(package.__path__, package_name + "."):
                    try:
                        modules.append(importlib.import_module(info.name))
                    except Exception as exc:  # noqa: BLE001
                        self.errors.append({"module": info.name, "error": f"{type(exc).__name__}: {exc}"})
            for module in modules:
                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, Node) and obj is not Node and obj.__module__ == module.__name__
                            and not inspect.isabstract(obj) and not obj.__name__.startswith("_")):
                        known = self.classes.get(obj.__name__)
                        if known is not None and known is not obj:
                            self.errors.append({"module": module.__name__,
                                                "error": f"duplicate node class name {obj.__name__!r}"})
                            continue
                        self.classes[obj.__name__] = obj
        self._loaded = True

    def settings_registry(self, extra: Iterable[SettingSpec] = ()) -> SettingsRegistry:
        classes, _ = self.load()
        contributed = [spec for cls in classes.values() for spec in cls.SETTINGS]
        registry = SettingsRegistry(contributed, extra)
        return registry
