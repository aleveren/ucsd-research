import numpy as np
import pandas as pd
import re

def row_to_spectrum(row, columns = None):
    if columns is None:
        assert isinstance(row, pd.Series)
        columns = row.index
    regex = re.compile(r'wavelength_(\d+\.?\d*)')
    wavelengths = []
    intensities = []
    for i, colname in enumerate(columns):
        regex_result = regex.search(colname)
        if regex_result:
            wavelengths.append(float(regex_result.group(1)))
            intensities.append(row[i])
    return np.array(wavelengths), np.array(intensities)
