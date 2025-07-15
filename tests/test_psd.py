import numpy as np
import pytest
from spectral_analysis_tools import PSD


def test_psd_output_shapes():
    rng = np.random.default_rng(0)
    M, N = 3, 128
    data = rng.normal(size=(M, N))
    result = PSD(data, dt=1.0, Ndist=10, verbose=False)

    assert 'freq' in result
    assert 'Lambda' in result
    assert 'Lambda_ci' in result

    nfreq = N // 2 + 1
    assert result['freq'].shape == (nfreq,)
    assert result['Lambda'].shape == (nfreq,)
    assert result['Lambda_ci'].shape == (nfreq, 2)
