import pandas as pd
import json
from peaks_util import extract_peaks, label_peaks
from data_util import row_to_spectrum
from filter_elements import filter_elements

def label_spectrum_single(
        row,
        known_emission_lines,
        peak_threshold_fraction,
        label_peaks_strategy,
        filter_elements_fraction,
        delta):

    xs, ys = row_to_spectrum(row)
    px, py = extract_peaks(xs, ys,
        threshold_fraction = peak_threshold_fraction,
        delta = delta)
    filtered_elements = filter_elements(px, known_emission_lines,
        drop_if_below_fraction = filter_elements_fraction,
        delta = delta,
        group_by = ["elt", "ex"])
    e_to_p, p_to_e, unlabeled = label_peaks(
        px,
        known_emission_lines,
        delta = delta,
        strategy = label_peaks_strategy)

    labeled_peaks = json.dumps(p_to_e)
    extra_columns = pd.Series([labeled_peaks], index = ['labeled_peaks'])
    return row.append(extra_columns)

def label_spectra(df, *args, **kwargs):
    return df.apply(label_spectrum_single,
        axis = 'columns',
        args = tuple(args),
        **kwargs)
