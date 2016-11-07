from peaks_util import (nearby_peaks, nearby_peaks_slow, extract_peaks,
    label_peaks_parsimonious, label_peaks_verbose)
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal

w_test = [10, 20, 30, 30, 40, 40, 50, 60]

known_emission_lines = pd.DataFrame({
    "wav_mars": [10., 20., 20., 30.],
    "elt":      ["A", "A", "B", "C"]})

def test_nearby_peaks():
    assert nearby_peaks(w_test, 34, delta = 9) == (2, 6)
    assert nearby_peaks(w_test, 34, delta = 5) == (2, 4)
    assert nearby_peaks(w_test, 34, delta = 3) == (4, 4)
    assert nearby_peaks(w_test, 25, delta = 7) == (1, 4)
    assert nearby_peaks(w_test, 45, delta = 7) == (4, 7)
    assert nearby_peaks(w_test, 55, delta = 7) == (6, 8)
    assert nearby_peaks(w_test, 35, delta = 100) == (0, 8)
    assert nearby_peaks(w_test, 5,  delta = 100) == (0, 8)
    assert nearby_peaks(w_test, 5,  delta = 3) == (0, 0)
    assert nearby_peaks(w_test, 65, delta = 3) == (8, 8)

def test_nearby_peaks_slow():
    assert nearby_peaks_slow(w_test, 34, delta = 9) == (2, 6)
    assert nearby_peaks_slow(w_test, 34, delta = 5) == (2, 4)
    assert nearby_peaks_slow(w_test, 34, delta = 3) == (4, 4)
    assert nearby_peaks_slow(w_test, 25, delta = 7) == (1, 4)
    assert nearby_peaks_slow(w_test, 45, delta = 7) == (4, 7)
    assert nearby_peaks_slow(w_test, 55, delta = 7) == (6, 8)
    assert nearby_peaks_slow(w_test, 35, delta = 100) == (0, 8)
    assert nearby_peaks_slow(w_test, 5,  delta = 100) == (0, 8)
    assert nearby_peaks_slow(w_test, 5,  delta = 3) == (0, 0)
    assert nearby_peaks_slow(w_test, 65, delta = 3) == (8, 8)

def test_extract_peaks():
    x = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
    y = np.array([1., 8., 1., 9., 8., 9., 1., 1., 5., 1., 1., 4., 1.])
    px, py = extract_peaks(x, y, threshold_fraction = 0.5, delta = 0.25)
    assert_array_equal(px, [0.3, 0.8])
    assert_array_equal(py, [9.0, 5.0])

    px, py = extract_peaks(x, y, threshold_fraction = 0.5, delta = 0.01)
    assert_array_equal(px, [0.1, 0.3, 0.5, 0.8])
    assert_array_equal(py, [8.0, 9.0, 9.0, 5.0])

    px, py = extract_peaks(x, y, threshold_fraction = 0.2, delta = 0.01)
    assert_array_equal(px, [0.1, 0.3, 0.5, 0.8, 1.1])
    assert_array_equal(py, [8.0, 9.0, 9.0, 5.0, 4.0])

def test_label_peaks_parsimonious():
    px = [10.1, 20.1, 30.1, 40.1]
    e_to_p, p_to_e, unlabeled = label_peaks_parsimonious(
        px, known_emission_lines, delta = 0.25)
    assert set(e_to_p.keys()) == set(["A", "C"])
    assert e_to_p["A"] == [10.1, 20.1]
    assert e_to_p["C"] == [30.1]
    assert set(p_to_e.keys()) == set([10.1, 20.1, 30.1])
    assert p_to_e[10.1] == ["A"]
    assert p_to_e[20.1] == ["A"]
    assert p_to_e[30.1] == ["C"]
    assert unlabeled == [40.1]

def test_label_peaks_verbose():
    px = [10.1, 20.1, 30.1, 40.1]
    e_to_p, p_to_e, unlabeled = label_peaks_verbose(
        px, known_emission_lines, delta = 0.25)
    assert set(e_to_p.keys()) == set(["A", "B", "C"])
    assert e_to_p["A"] == [10.1, 20.1]
    assert e_to_p["B"] == [20.1]
    assert e_to_p["C"] == [30.1]
    assert set(p_to_e.keys()) == set([10.1, 20.1, 30.1])
    assert p_to_e[10.1] == ["A"]
    assert p_to_e[20.1] == ["A", "B"]
    assert p_to_e[30.1] == ["C"]
    assert unlabeled == [40.1]
