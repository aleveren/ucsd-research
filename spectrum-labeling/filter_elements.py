from peaks_util import nearby_peaks
from collections import defaultdict
import numpy as np

def filter_elements(
        observed_peak_wavelengths,
        known_emission_lines,
        drop_if_below_fraction,
        delta,
        group_by):

    peaks_grouped = defaultdict(set)
    for row_index, row in known_emission_lines.iterrows():
        key = tuple([row[var] for var in group_by])
        peaks_grouped[key].add(row["wav_mars"])

    overall_mask_to_drop = np.zeros(len(known_emission_lines), dtype='bool')
    for key, emission_wavs in peaks_grouped.items():
        denom = 0
        numer = 0
        for w in emission_wavs:
            denom += 1
            i_lo, i_hi = nearby_peaks(
                observed_peak_wavelengths, w, delta = delta)
            nearby = observed_peak_wavelengths[i_lo : i_hi]
            if len(nearby) > 0:
                numer += 1
        fraction = numer / float(denom)
        if fraction < drop_if_below_fraction:
            mask = np.ones(len(known_emission_lines), dtype='bool')
            for var_index, var in enumerate(group_by):
                mask &= (known_emission_lines[var] == key[var_index])
            overall_mask_to_drop |= mask

    omitted = known_emission_lines[overall_mask_to_drop]
    filtered = known_emission_lines[~overall_mask_to_drop]

    assert len(omitted) + len(filtered) == len(known_emission_lines)
    
    return filtered
