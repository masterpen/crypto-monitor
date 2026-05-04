import tempfile
import os
from pathlib import Path
from src.data.utils import _safe_write_csv, wipe_cache, cache_stats
import pandas as pd


def test_safe_write_csv():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.csv"
        df = pd.DataFrame({"a": [1, 2, 3]})
        _safe_write_csv(df, path)
        assert path.exists()
        loaded = pd.read_csv(path)
        assert len(loaded) == 3


def test_wipe_cache_by_symbol_and_interval():
    with tempfile.TemporaryDirectory() as tmp:
        import src.data.utils as ut
        original_dir = ut._CACHE_DIR
        ut._CACHE_DIR = Path(tmp)
        try:
            p = ut._cache_path("TESTUSDT", "1h")
            p.parent.mkdir(exist_ok=True)
            _safe_write_csv(pd.DataFrame({"x": [1]}), p)
            assert p.exists()
            ut.wipe_cache("TESTUSDT", "1h")
            assert not p.exists()
        finally:
            ut._CACHE_DIR = original_dir


def test_wipe_cache_all():
    with tempfile.TemporaryDirectory() as tmp:
        import src.data.utils as ut
        original_dir = ut._CACHE_DIR
        ut._CACHE_DIR = Path(tmp)
        try:
            p1 = ut._cache_path("AUSDT", "1h")
            p2 = ut._cache_path("BUSDT", "4h")
            p1.parent.mkdir(exist_ok=True)
            _safe_write_csv(pd.DataFrame({"x": [1]}), p1)
            _safe_write_csv(pd.DataFrame({"x": [1]}), p2)
            ut.wipe_cache()
            assert not p1.exists()
            assert not p2.exists()
        finally:
            ut._CACHE_DIR = original_dir


def test_cache_stats_empty():
    with tempfile.TemporaryDirectory() as tmp:
        import src.data.utils as ut
        original_dir = ut._CACHE_DIR
        ut._CACHE_DIR = Path(tmp)
        try:
            stats = ut.cache_stats()
            assert stats['files'] == 0
            assert stats['total_size_mb'] == 0
        finally:
            ut._CACHE_DIR = original_dir