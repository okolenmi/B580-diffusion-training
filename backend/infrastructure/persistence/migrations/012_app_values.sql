-- What a person last set on an app's page, kept apart from the published document so that using an app never makes a version:
-- {"nodes": {uid: {port: value}}, "memory": {key: value}} (the shape a run is given), NULL = nothing changed since the last save.
ALTER TABLE apps ADD COLUMN values_json TEXT;
