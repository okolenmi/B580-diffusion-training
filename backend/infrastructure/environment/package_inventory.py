"""Installed packages read from dist-info, without importing them."""

from __future__ import annotations

from ...application.ports.environment import InstalledPackage, PackageInventory


class MetadataPackageInventory(PackageInventory):
    """`importlib.metadata` -- reads dist-info, imports nothing.

    Verified on this machine that asking for torch's version does not put
    torch in `sys.modules`, which is the property the whole design rests
    on: the server must be able to report a missing torch without having
    loaded one.
    """

    def installed(self) -> tuple[InstalledPackage, ...]:
        import importlib.metadata as metadata

        found: list[InstalledPackage] = []
        seen: set[str] = set()
        for dist in metadata.distributions():
            name = dist.metadata["Name"]
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            found.append(
                InstalledPackage(
                    distribution=name,
                    import_name=self.import_name_for(name),
                    version=dist.version or "",
                )
            )
        return tuple(sorted(found, key=lambda p: p.distribution.lower()))

    def version_of(self, distribution: str) -> str | None:
        import importlib.metadata as metadata

        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            return None
