import numpy as np
import pandas as pd
from data_util import row_to_spectrum

def test_row_to_spectrum():
    intensities = [10., 20., 30., 40., 50.]
    columns = ["x", "wavelength_1.2", "y", "wavelength_2.3", "z"]
    s = pd.Series(intensities, index = columns)
    x, y = row_to_spectrum(s)
    assert np.array_equal(x, [1.2, 2.3])
    assert np.array_equal(y, [20., 40.])

    x, y = row_to_spectrum(intensities, columns = columns)
    assert np.array_equal(x, [1.2, 2.3])
    assert np.array_equal(y, [20., 40.])
