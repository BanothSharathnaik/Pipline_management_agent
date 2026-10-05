import numpy as np
import pandas as pd


def test_numpy_works():
    v = np.array([3.0, 4.0])
    assert float(np.linalg.norm(v)) == 5.0


def test_pandas_works():
    df = pd.DataFrame({"status": ["SUCCESS", "FAILED"]})
    assert len(df) == 2


def test_src_importable():
    import src.monitoring.models  # noqa: F401