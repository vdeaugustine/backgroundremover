"""User preference JSON persistence."""

import json
import os

_PREFS_DIR = os.path.join(os.path.expanduser("~"), ".config", "backgroundremover")
_PREFS_FILE = os.path.join(_PREFS_DIR, "preferences.json")

PREF_KEYS = (
    "image_cleanup_threshold",
    "video_cleanup_threshold",
    "deduplication_threshold",
)


def load_preferences():
    """Return a dict of saved user preferences, or an empty dict if none exist."""
    try:
        with open(_PREFS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_preferences(prefs: dict):
    """Persist a dict of user preferences to disk."""
    try:
        os.makedirs(_PREFS_DIR, exist_ok=True)
        with open(_PREFS_FILE, "w", encoding="utf-8") as fh:
            json.dump(prefs, fh, indent=2)
    except Exception:
        pass
