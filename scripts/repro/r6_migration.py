import sys, tempfile, sqlite3
from pathlib import Path
sys.path.insert(0,__import__("os").environ.get("REPO","."))
import backend.infrastructure.persistence.sqlite as S
with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp); mig = tmp/"migrations"; mig.mkdir()
    (mig/"001_a.sql").write_text("CREATE TABLE a (id INTEGER);\nCREATE TABLE b (id INTEGER);\nINSERT INTO nonexistent VALUES (1);\n")  # fails at stmt 3
    # point the runner at our dir
    orig = S.Path
    db = S.SqliteDatabase(tmp/"x.db")
    real_file = S.__file__
    class P(type(Path())): pass
    S.__file__ = str(tmp/"persistence"/"sqlite.py"); (tmp/"persistence").mkdir(); (tmp/"persistence"/"migrations").symlink_to(mig)
    for attempt in (1,2):
        try: db.initialize(); print(f"start #{attempt}: ok")
        except Exception as e: print(f"start #{attempt}: FAILED -> {type(e).__name__}: {e}")
    c = sqlite3.connect(tmp/"x.db"); print("tables left behind by the failed migration:", [r[0] for r in c.execute("select name from sqlite_master where type='table' and name in ('a','b')")])
