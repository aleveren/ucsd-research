import numpy as np
from set_cover import set_cover_approx_fast
from scipy.signal import argrelmax
from collections import defaultdict, OrderedDict

def nearby_peaks_slow(wavelengths, target_wavelength, delta):
    '''
    Find indices of wavelengths that are within delta of the target wavelength
    (Slow implementation for comparison / testing only)
    '''
    wavelengths = np.asarray(wavelengths)
    if wavelengths[-1] < target_wavelength - delta:
        i_lo = len(wavelengths)
    else:
        i_lo = min([i for i in xrange(len(wavelengths))
            if wavelengths[i] >= target_wavelength - delta])
    if wavelengths[0] > target_wavelength + delta:
        i_hi = 0
    else:
        i_hi = max([i+1 for i in xrange(len(wavelengths))
            if wavelengths[i] <= target_wavelength + delta])
    return i_lo, i_hi

def nearby_peaks(wavelengths, target_wavelength, delta):
    '''
    Find indices of wavelengths that are within delta of the target wavelength
    '''
    wavelengths = np.asarray(wavelengths)
    lo = target_wavelength - delta
    hi = target_wavelength + delta

    wav_prev = lambda i: wavelengths[i - 1] if i > 0 else -np.inf
    wav_next = lambda i: wavelengths[i] if i < len(wavelengths) else np.inf
    
    def find_lo_index(i_min, i_max):
        i_pivot = int((i_min + i_max) / 2)
        if i_min == i_max:
            result = i_pivot
        elif wav_next(i_pivot) < lo:
            result = find_lo_index(i_pivot + 1, i_max)
        elif wav_prev(i_pivot) < lo:
            result = i_pivot
        else:
            result = find_lo_index(i_min, i_pivot)
        return result

    def find_hi_index(i_min, i_max):
        i_pivot = int((i_min + i_max) / 2)
        if i_min == i_max:
            result = i_pivot
        elif wav_prev(i_pivot) > hi:
            result = find_hi_index(i_min, i_pivot)
        elif wav_next(i_pivot) > hi:
            result = i_pivot
        else:
            result = find_hi_index(i_pivot + 1, i_max)
        return result
        
    i_lo = find_lo_index(0, len(wavelengths))
    i_hi = find_hi_index(0, len(wavelengths))

    return i_lo, i_hi

def extract_peaks(xs, ys, threshold_fraction, delta):
    '''
    Given lists of x and y coordinates, find all local maxima where
    y is above `threshold_fraction * max(ys)` and x is not within
    +/-delta of a higher local maximum
    '''
    thresh = max(ys) * threshold_fraction
    max_indices = argrelmax(ys)[0]
    max_indices = max_indices[ys[max_indices] > thresh]
    peak_xs = []
    peak_ys = []
    for i in max_indices:
        if len(peak_xs) == 0:
            peak_xs.append(xs[i])
            peak_ys.append(ys[i])
        elif abs(peak_xs[-1] - xs[i]) > delta:
            peak_xs.append(xs[i])
            peak_ys.append(ys[i])
        elif len(peak_xs) > 0 and ys[i] > peak_ys[-1]:
            peak_xs[-1] = xs[i]
            peak_ys[-1] = ys[i]
    return np.array(peak_xs), np.array(peak_ys)

def label_peaks_verbose(peaks, known_emission_lines, delta):
    '''
    Label each peak according to the set of elements with
    known emission lines within +/-delta of the peak
    '''
    known_emission_lines = known_emission_lines.sort_values(by="wav_mars")
    known_wavelengths = known_emission_lines["wav_mars"]
    e_to_p = defaultdict(list)
    p_to_e = defaultdict(list)
    unlabeled = []
    for p in peaks:
        i_lo, i_hi = nearby_peaks(known_wavelengths, p, delta)
        elts = sorted(list(set(known_emission_lines["elt"].iloc[i_lo:i_hi])))
        for e in elts:
            e_to_p[e].append(p)
            p_to_e[p].append(e)
        if len(elts) == 0:
            unlabeled.append(p)
    return e_to_p, p_to_e, unlabeled

def label_peaks_parsimonious(peaks, known_emission_lines, delta):
    '''
    Out of all the peaks that are within +/-delta of at least one
    known emission line, find a minimal set of elements
    (via unweighted greedy set cover) that "explains" all such peaks
    '''
    known_emission_lines = known_emission_lines.sort_values(by="wav_mars")
    peaks = sorted(peaks)
    elts_to_peak_indices = defaultdict(set)
    for row_index, row in known_emission_lines.iterrows():
        elt = row["elt"]
        i_lo, i_hi = nearby_peaks(peaks, row["wav_mars"], delta = delta)
        for i in xrange(i_lo, i_hi):
            elts_to_peak_indices[elt].add(i)
    sets = []
    elts = sorted(list(np.unique(known_emission_lines["elt"])))
    for elt in elts:
        sets.append(list(elts_to_peak_indices[elt]))

    best_cover = set_cover_approx_fast(sets)

    cover_elts_to_peaks = OrderedDict()
    cover_peaks_to_elts = OrderedDict()
    unlabeled = set(peaks)
    for i in best_cover:
        elt = elts[i]
        for peak_index in elts_to_peak_indices[elt]:
            peak = peaks[peak_index]
            if elt not in cover_elts_to_peaks:
                cover_elts_to_peaks[elt] = []
            cover_elts_to_peaks[elt].append(peak)
            if peak not in cover_peaks_to_elts:
                cover_peaks_to_elts[peak] = []
            cover_peaks_to_elts[peak].append(elt)
            unlabeled -= set([peak])

    return cover_elts_to_peaks, cover_peaks_to_elts, sorted(list(unlabeled))
