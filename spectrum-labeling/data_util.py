import numpy as np
import pandas as pd
import re

def row_to_spectrum(row, columns = None, gap_insert_nan = None):
    if columns is None:
        assert isinstance(row, pd.Series)
        columns = row.index
    regex = re.compile(r'wavelength_(\d+\.?\d*)')
    wavelengths = []
    intensities = []
    for i, colname in enumerate(columns):
        regex_result = regex.search(colname)
        if regex_result:
            w = float(regex_result.group(1))
            y = row[i]
            if gap_insert_nan is not None and len(wavelengths) > 0 \
                    and (w - wavelengths[-1]) > gap_insert_nan:
                wavelengths.append(np.nan)
                intensities.append(np.nan)
            wavelengths.append(w)
            intensities.append(y)
    return np.array(wavelengths), np.array(intensities)
