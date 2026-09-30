-- Server-level settings: a small key/value table (keys documented on
-- application.ports.settings_store.SETTINGS_KEYS). Absent row = unset.
CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
