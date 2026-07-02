"""Legacy compatibility module — re-exports from the new config_model and config_io.

Previously this contained TrainingConfig with inline UI metadata (OptionMeta),
TOML I/O with regex-based saving, and the load_config() → argparse.Namespace path.
All of that has been moved to:
  - converter/config_model.py — pure Pydantic models with discriminated unions
  - converter/config_io.py    — TOML I/O with flat-to-section migration
  - server/config_ui.py       — UI metadata definitions (separate from models)

Use the new modules directly for new code.
"""

from .config_model import TrainingConfig
from .config_io import read_config as load_config, write_default_config

# Legacy constant — kept for backward compat
DEFAULTS = TrainingConfig().model_dump(exclude_none=True)
