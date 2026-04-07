"""GUI package: call configure_bundle_paths before any torch use via gui.device or gui.app."""

from gui.runtime import configure_bundle_paths

configure_bundle_paths()

__all__ = ["configure_bundle_paths"]
