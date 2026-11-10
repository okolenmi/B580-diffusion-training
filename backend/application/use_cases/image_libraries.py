"""The Datasets page's operations over image libraries; each is a thin step over the ``ImageLibraries`` port."""

from __future__ import annotations

from pathlib import Path

from ..ports.image_libraries import ImageLibraries


class _UseCase:
    def __init__(self, libraries: ImageLibraries) -> None:
        self._libraries = libraries


class ListLibraries(_UseCase):
    def execute(self) -> list[dict]:
        return self._libraries.list()


class CreateLibrary(_UseCase):
    def execute(self, name: str) -> dict:
        return self._libraries.create(name.strip())


class DeleteLibrary(_UseCase):
    def execute(self, name: str) -> None:
        self._libraries.delete(name)


class ListImages(_UseCase):
    def execute(self, name: str, *, set_name: str, query: str, offset: int, limit: int) -> dict:
        return self._libraries.page(name, set_name=set_name, query=query, offset=offset, limit=limit)


class ReadImage(_UseCase):
    def execute(self, name: str, image_id: str, thumb: int | None) -> tuple[bytes, str]:
        return self._libraries.image(name, image_id, thumb=thumb)


class EditCaptions(_UseCase):
    def execute(self, name: str, **change) -> int:
        return self._libraries.edit_captions(name, **change)


class RemoveImages(_UseCase):
    def execute(self, name: str, ids: list[str], set_name: str | None) -> int:
        return self._libraries.remove(name, ids, set_name)


class AddToSet(_UseCase):
    def execute(self, name: str, set_name: str, ids: list[str]) -> int:
        return self._libraries.add_to_set(name, set_name.strip(), ids)


class DeleteSet(_UseCase):
    def execute(self, name: str, set_name: str) -> None:
        self._libraries.delete_set(name, set_name)


class DeleteCache(_UseCase):
    def execute(self, name: str, vae: str) -> None:
        self._libraries.delete_cache(name, vae)


class StartImport(_UseCase):
    def execute(self, name: str, folder: str, *, recursive: bool, set_name: str) -> dict:
        return self._libraries.start_import(name, Path(folder.strip()).expanduser(), recursive=recursive, set_name=set_name.strip() or "all")


class ListImports(_UseCase):
    def execute(self, name: str) -> list[dict]:
        return self._libraries.imports(name)
