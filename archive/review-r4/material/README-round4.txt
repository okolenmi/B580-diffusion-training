Round-4 reproductions (run from the repo root; COMFY_DIR must point at any empty dir)
r17_readiness_fanout.py        N4-01  8 concurrent readiness probes -> 8 simultaneous torch-importing processes
r18_sweep_deletes_crash_log.py N4-02  failed run's log (the only copy of the traceback) is deleted by the startup sweep
N4-03: env -u COMFY_DIR python3 backend/tests/test_config.py   (also test_installer.py, test_settings.py)
Note: tests that spawn a real child need Python >= 3.14 (the project enforces it).
